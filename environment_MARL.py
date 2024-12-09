## TODO
# add end condition, no rewards in x steps?

## add skills
# basic skill actions
# modifiers
# more more more
## add skill picking
# picking round
## update ActorCritic
# LSTM
# action set
# dynamic skill features
## update training
# opponent policy updates
# opponent policy buffer
# wandb logging
# logging and performance / loss analysis
# 

## refactor

# environment_MARL.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import chex
from flax import struct
from typing import Tuple, Dict
from functools import partial
from gymnax.environments.spaces import Discrete, Box
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jax.experimental import host_callback

import data_classes
import ability_actions
from utils import euclidean_distance, generate_unique_pairs, get_latest_checkpoint_dir
import base_actions
from reward import reward_function
from config import train_config, env_config
# from data_classes import UnitState, EnvState, EnvParams
from render import render_game_state

# Dynamic import of action functions
base_action_functions = []
for create_fn in base_actions.base_action_registry.values():
    base_action_functions.extend(create_fn())
base_action_fns = [action.execute for action in base_action_functions]

ability_action_functions = []
for create_fn in ability_actions.ability_registry.values():
    ability_action_functions.extend(create_fn())
ability_action_fns = [action.execute for action in ability_action_functions]

# Update num_actions
num_base_actions = len(base_action_functions)
num_abilities = len(ability_action_functions)

checkpoint_dir = '/home/jvnheagerty/checkpoints'

schema_classes = data_classes.create_struct_dataclass(data_classes.schema)
AbilityState = schema_classes['AbilityState']
AbilityStatusState = schema_classes['AbilityStatusState']
UnitState = schema_classes['UnitState']
GameState = schema_classes['GameState']

low, high = data_classes.get_observation_bounds(data_classes.schema)

class RL_Roguelike_JAX_MARL(MultiAgentEnv):

    def __init__(self,
                 num_agents=2,
                 agents=None,
                 action_spaces=None,
                 observation_spaces=None,
                 obs_size=None,
                 num_moves=None,
                 ):
        super().__init__(num_agents)
        # Store ability parameters as static arrays
        self.ability_params = jnp.array([[
            ability.base_cooldown,
            ability.parameter_1,
            ability.parameter_2,
            ability.parameter_3
        ] for ability in ability_action_functions])
        
        self.num_agents = num_agents
        self.agent_range = jnp.arange(num_agents)

        if agents is None:
            self.agents = [f"agent_{i}" for i in range(num_agents)]
        else:
            assert len(
                agents) == num_agents, f"Number of agents {len(agents)} does not match number of agents {num_agents}"
            self.agents = agents

        self.num_moves = num_base_actions + 2  # Add 1 for the ability slot TODO: MAKE DYNAMIC

        # TODO remove num_moves? no longer append available?
        if obs_size is None:
            self.obs_size = (data_classes.get_observation_size(data_classes.schema))#+self.num_moves)

        self.action_set = jnp.arange(self.num_moves)
        if action_spaces is None:
            self.action_spaces = {i: Discrete(self.num_moves) for i in self.agents}
        if observation_spaces is None:
            self.observation_spaces = {i: Box(low, high, (len(low),), jnp.float32) for i in self.agents}#(18+num_actions,), jnp.float32) for i in self.agents}

    def get_legal_moves(self, state: GameState) -> chex.Array:
        """Get all agents' legal moves"""

        @partial(jax.vmap, in_axes=[0, None])
        def _legal_moves(aidx: int, state: GameState) -> chex.Array:
            return jax.lax.cond(
                aidx == 0,
                lambda _: self.get_available_actions(state.player, state.enemy, state),
                lambda _: self.get_available_actions(state.enemy, state.player, state),
                operand=None
                )

        legal_moves = _legal_moves(self.agent_range, state)

        return {a: legal_moves[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, GameState]:
        """Reset the environment."""
        key, key_pos, key_abilities = jax.random.split(key, 3)
        
        # Generate positions
        x1, y1, x2, y2 = generate_unique_pairs(key_pos)
        initial_distance = jnp.float32(euclidean_distance(x1, y1, x2, y2))

        ability_idx1, ability_idx2, pick_pool_0, pick_pool_1, pick_pool_2 = jax.random.choice(key, num_abilities, shape=(5,), replace=False)
        # pick_pool = tuple(pick_pool)

        # Get ability parameters using vmap
        def get_ability_params(idx):
            return jax.lax.dynamic_slice(self.ability_params, (idx, 0), (1, 4))[0]

        player_params = get_ability_params(ability_idx1)
        enemy_params = get_ability_params(ability_idx2)
        pool_ability_1_params = get_ability_params(pick_pool_0)
        pool_ability_2_params = get_ability_params(pick_pool_1)
        pool_ability_3_params = get_ability_params(pick_pool_2)

        # Create ability states
        player_ability = AbilityState(
            ability_index=jnp.int32(ability_idx1),
            base_cooldown=jnp.int32(player_params[0]),
            current_cooldown=jnp.int32(0),
            parameter_1=jnp.float32(player_params[1]),
            parameter_2=jnp.float32(player_params[2]),
            parameter_3=jnp.float32(player_params[3])
        )
        
        enemy_ability = AbilityState(
            ability_index=jnp.int32(ability_idx2),
            base_cooldown=jnp.int32(enemy_params[0]),
            current_cooldown=jnp.int32(0),
            parameter_1=jnp.float32(enemy_params[1]),
            parameter_2=jnp.float32(enemy_params[2]),
            parameter_3=jnp.float32(enemy_params[3])
        )
        
        pool_ability_1 = AbilityState(
            ability_index=jnp.int32(pick_pool_0),
            base_cooldown=jnp.int32(pool_ability_1_params[0]),
            current_cooldown=jnp.int32(0),
            parameter_1=jnp.float32(pool_ability_1_params[1]),
            parameter_2=jnp.float32(pool_ability_1_params[2]),
            parameter_3=jnp.float32(pool_ability_1_params[3])
        )
        
        pool_ability_2 = AbilityState(
            ability_index=jnp.int32(pick_pool_1),
            base_cooldown=jnp.int32(pool_ability_2_params[0]),
            current_cooldown=jnp.int32(0),
            parameter_1=jnp.float32(pool_ability_2_params[1]),
            parameter_2=jnp.float32(pool_ability_2_params[2]),
            parameter_3=jnp.float32(pool_ability_2_params[3])
        )
        
        pool_ability_3 = AbilityState(
            ability_index=jnp.int32(pick_pool_2),
            base_cooldown=jnp.int32(pool_ability_3_params[0]),
            current_cooldown=jnp.int32(0),
            parameter_1=jnp.float32(pool_ability_3_params[1]),
            parameter_2=jnp.float32(pool_ability_3_params[2]),
            parameter_3=jnp.float32(pool_ability_3_params[3])
        )

        # Create states
        player = data_classes.create_unit_state(
            UnitState,
            {'unit_id': jnp.int32(1),
             'location_x': x1,
             'location_y': y1,
             'ability_state_1': player_ability
            })

        enemy = data_classes.create_unit_state(
            UnitState,
            {'unit_id': jnp.int32(-1),
             'location_x': x2,
             'location_y': y2,
             'ability_state_1': enemy_ability
            })

        # Create initial state without available actions
        state = GameState(
            player=player,
            enemy=enemy,
            distance_to_enemy=initial_distance,
            steps=jnp.int32(0),
            turn_count=jnp.int32(0),
            previous_closest_distance=initial_distance,
            initial_distance=initial_distance,
            cur_player_idx=jnp.zeros(self.num_agents).at[0].set(1),
            terminal=False,
            pool_ability_1=pool_ability_1,
            pool_ability_2=pool_ability_2,
            pool_ability_3=pool_ability_3,
            pick_mode=1,
            pool_ability_1_picked=jnp.int32(0),
            pool_ability_2_picked=jnp.int32(0),
            pool_ability_3_picked=jnp.int32(0),
            pick_count=jnp.int32(0),
        )

        # Update available actions using lax.switch based validation
        available_actions_player = self.get_available_actions(state.player, state.enemy, state)
        available_actions_enemy = self.get_available_actions(state.enemy, state.player, state)
        
        state = state.replace(
            player=player.replace(available_actions=available_actions_player),
            enemy=enemy.replace(available_actions=available_actions_enemy)
        )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: GameState) -> Dict:
        """Get all agents' observations using the schema definition"""
        
        @partial(jax.vmap, in_axes=[0, None])
        def _observation(aidx: int, state: GameState) -> chex.Array:
            return data_classes.make_schema_based_observation(state, aidx, data_classes.schema)

        obs = _observation(self.agent_range, state)
        return {a: obs[i] for i, a in enumerate(self.agents)}
    
    def function_mapper(self, key: chex.PRNGKey, state: GameState, action: int, unit, target):
        """Handle both ability slots"""
        def do_base_action(key: chex.PRNGKey):
            return lax.switch(action, base_action_fns, key, state, unit, target, jnp.int32(-1))
            
        def do_ability_1_action(key: chex.PRNGKey):
            return lax.switch(unit.ability_state_1.ability_index, 
                            ability_action_fns, key, state, unit, target, jnp.int32(0))
                            
        def do_ability_2_action(key: chex.PRNGKey):
            return lax.switch(unit.ability_state_2.ability_index, 
                            ability_action_fns, key, state, unit, target, jnp.int32(1))
        
        key, key_ = jax.random.split(key)
        
        return lax.cond(
            action < num_base_actions,
            lambda _: do_base_action(key_),
            lambda _: lax.cond(
                action == num_base_actions,
                lambda _: do_ability_1_action(key_),
                lambda _: do_ability_2_action(key_),
                operand=None
            ),
            operand=None
        )
        
        # Update available actions
        new_state = new_state.replace(
            player=new_state.player.replace(
                available_actions=self.get_available_actions(new_state.player, new_state.enemy, new_state)
            ),
            enemy=new_state.enemy.replace(
                available_actions=self.get_available_actions(new_state.enemy, new_state.player, new_state)
            )
        )
        
        return new_state

    def get_available_actions(self, unit, target, state: GameState) -> chex.Array:
        base_actions_available = jnp.array(
            [action.is_valid(state, unit, target) for action in base_action_functions], 
            dtype=jnp.int32
        )
        
        def validate_ability(ability_idx, ability_slot, state, unit, target):
            def ability_case(i):
                return ability_action_functions[i].is_valid(state, unit, target)
                
            return lax.switch(ability_idx, 
                            [lambda: ability_case(i) for i in range(len(ability_action_functions))])
        
        ability_1_available = validate_ability(unit.ability_state_1.ability_index, 0, state, unit, target)
        ability_2_available = validate_ability(unit.ability_state_2.ability_index, 1, state, unit, target)
        
        return jnp.concatenate([
            base_actions_available,
            jnp.array([ability_1_available, ability_2_available], dtype=jnp.int32)
        ])

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: GameState, actions: Dict,
                 ) -> Tuple[chex.Array, GameState, Dict, Dict, Dict]:
        """
        Step the environment
        Executes one turn
        """
        # get actions as array
        actions = jnp.array([actions[i] for i in self.agents])
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        action = actions.at[aidx].get()
        action = action[0]# - 1 # ignore noop for acting agent

        old_state = state

        # perform player action and resolve the new state
        key, key_ = jax.random.split(key)
        new_state = lax.cond(
                jnp.equal(aidx, 0),
                lambda _: self.function_mapper(key_, old_state, action, old_state.player, old_state.enemy),
                lambda _: self.function_mapper(key_, old_state, action, old_state.enemy, old_state.player),
                operand = None
                )

        player_reward, enemy_reward = reward_function(old_state, new_state, action)

        # check enemy dead
        enemy_dead = state.enemy.health_current <= 0
        # check player dead
        player_dead = state.player.health_current <= 0
        # Check number of steps in episode termination condition
        done_steps = state.steps >= env_config['MAX_STEPS']

        done = jnp.logical_or(jnp.logical_or(enemy_dead, player_dead), done_steps)

        new_state = new_state.replace(terminal = done)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        rewards = {self.agents[0]: player_reward,
                   self.agents[1]: enemy_reward,
                   }
        rewards["__all__"] = 0 # TODO do we need this?? what does it do??

        info = {}

        return (
            lax.stop_gradient(self.get_obs(new_state)),
            lax.stop_gradient(new_state),
            rewards,
            dones,
            info
        )

    def terminal(self, state: GameState) -> bool:
        """Check whether state is terminal."""
        return state.terminal

    @property
    def name(self) -> str:
        """Environment name."""
        return "RL-Roguelike-JAX-MARL"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.num_moves

    def observation_space(self, agent: str):
        """ Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """ Action space for a given agent."""
        return self.action_spaces[agent]