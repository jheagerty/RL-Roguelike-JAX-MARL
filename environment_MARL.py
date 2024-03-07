## TODO
# get_legal_moves x
# reset x
# get obs x
# step env x
# end turn switches current player x
# add end condition, no rewards in x steps?
# reward function per team x


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

from utils import euclidean_distance, generate_unique_pairs, get_latest_checkpoint_dir
import base_actions
from reward import reward_function
from config import train_config, env_config
# from data_classes import UnitState, EnvState, EnvParams
from render import render_game_state

# Dynamic import of action functions
action_functions = []
for create_fn in base_actions.action_registry.values():
    action_functions.extend(create_fn())

# Update num_actions
num_actions = len(action_functions)

checkpoint_dir = '/home/jvnheagerty/checkpoints'

@struct.dataclass
class UnitState:
    unit_id: int
    health_current: float
    health_max: float
    location_x: int
    location_y: int
    melee_attack_base_damage: float
    ranged_attack_base_damage: float
    melee_attack_range: float
    ranged_attack_range: float
    movement_points_current: float
    movement_points_max: float
    action_points_current: float
    action_points_max: float
    available_actions: chex.Array

# @struct.dataclass
# class TeamState:
#     hero_1: UnitState
#     hero_2: UnitState
#     hero_3: UnitState # HeroState

@struct.dataclass
class State:
    # flatten this?
    # board: chex.Array
    player: UnitState
    enemy: UnitState
    distance_to_enemy: float
    steps: int
    turn_count: int
    previous_closest_distance: float
    initial_distance: float
    cur_player_idx: chex.Array
    terminal: bool

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

        self.num_agents = num_agents
        self.agent_range = jnp.arange(num_agents)

        if agents is None:
            self.agents = [f"agent_{i}" for i in range(num_agents)]
        else:
            assert len(
                agents) == num_agents, f"Number of agents {len(agents)} does not match number of agents {num_agents}"
            self.agents = agents

        if num_moves is None:
            self.num_moves = num_actions

        # TODO remove num_moves? no longer append available?
        if obs_size is None:
            self.obs_size = (18)#+self.num_moves) # TODO make dynamic

        self.action_set = jnp.arange(self.num_moves)
        if action_spaces is None:
            self.action_spaces = {i: Discrete(self.num_moves) for i in self.agents}
        if observation_spaces is None:
            low = [
                # 'board': spaces.Box(0, 1, (20, 20), jnp.int32),
                0, # 'player_health_current'
                0, # 'player_health_max'
                0, # 'player_location_x'
                0, # 'player_location_y'
                0, # 'player_melee_attack_base_damage'
                0, # 'player_ranged_attack_base_damage'
                0, # 'player_melee_attack_range'
                0, # 'player_ranged_attack_range'
                0, # 'player_movement_points_current'
                0, # 'player_movement_points_max'
                0, # 'player_action_points_current'
                0, # 'player_action_points_max'
                0, # 'enemy_health_current'
                0, # 'enemy_health_max'
                0, # 'enemy_location_x'
                0, # 'enemy_location_y'
                0, # 'distance_to_enemy'
                0, # 'turn_count'
            ]# + [0]*num_actions, # 'available_actions'

            high = [
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                19,
                19,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                19,
                19,
                jnp.finfo(jnp.float32).max,
                env_config['MAX_STEPS'],
            ]# + [1]*num_actions,
            self.observation_spaces = {i: Box(low, high, (18+num_actions,), jnp.float32) for i in self.agents}

    def get_legal_moves(self, state: State) -> chex.Array:
        """Get all agents' legal moves"""

        @partial(jax.vmap, in_axes=[0, None])
        def _legal_moves(aidx: int, state: State) -> chex.Array:
            return jax.lax.cond(
                aidx == 0,
                lambda _: self.get_available_actions(state.player, state.enemy, state),
                lambda _: self.get_available_actions(state.enemy, state.player, state),
                operand=None
                )

        legal_moves = _legal_moves(self.agent_range, state)

        return {a: legal_moves[i] for i, a in enumerate(self.agents)}

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, State]:
        """Reset the environment"""

        x1, y1, x2, y2 = generate_unique_pairs(key)
        initial_distance = jnp.float32(euclidean_distance(x1,x2,y1,y2))

        player = UnitState(
            unit_id = jnp.int32(1),
            health_current = jnp.float32(env_config['MAX_HEALTH']),
            health_max = jnp.float32(env_config['MAX_HEALTH']),
            location_x = x1,
            location_y = y1,
            melee_attack_base_damage = jnp.float32(env_config['MELEE_DAMAGE']),
            ranged_attack_base_damage = jnp.float32(env_config['RANGED_DAMAGE']),
            melee_attack_range = jnp.float32(env_config['MELEE_RANGE']),
            ranged_attack_range = jnp.float32(env_config['RANGED_RANGE']),
            movement_points_current = jnp.float32(env_config['MOVEMENT_POINTS']),
            movement_points_max = jnp.float32(env_config['MOVEMENT_POINTS']),
            action_points_current = jnp.float32(env_config['ACTION_POINTS']),
            action_points_max = jnp.float32(env_config['ACTION_POINTS']),
            available_actions = jnp.zeros(num_actions),
        )

        enemy = UnitState(
            unit_id = jnp.int32(-1),
            health_current = jnp.float32(env_config['MAX_HEALTH']),
            health_max = jnp.float32(env_config['MAX_HEALTH']),
            location_x = x2,
            location_y = y2,
            melee_attack_base_damage = jnp.float32(env_config['MELEE_DAMAGE']),
            ranged_attack_base_damage = jnp.float32(env_config['RANGED_DAMAGE']),
            melee_attack_range = jnp.float32(env_config['MELEE_RANGE']),
            ranged_attack_range = jnp.float32(env_config['RANGED_RANGE']),
            movement_points_current = jnp.float32(env_config['MOVEMENT_POINTS']),
            movement_points_max = jnp.float32(env_config['MOVEMENT_POINTS']),
            action_points_current = jnp.float32(env_config['ACTION_POINTS']),
            action_points_max = jnp.float32(env_config['ACTION_POINTS']),
            available_actions = jnp.zeros(num_actions),
        )

        state = State(
            player = player,
            enemy = enemy,
            distance_to_enemy = initial_distance,
            steps = jnp.int32(0),
            turn_count = jnp.int32(0),
            previous_closest_distance = initial_distance,
            initial_distance = initial_distance,
            cur_player_idx = jnp.zeros(self.num_agents).at[0].set(1), # TODO make random
            terminal = False,
        )
        
        state = state.replace(
            player = player.replace(available_actions = self.get_available_actions(state.player, state.enemy, state)),
            enemy = enemy.replace(available_actions = self.get_available_actions(state.enemy, state.player, state)),
            )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict:
        """
        Get all agents' observations
        """
        @partial(jax.vmap, in_axes=[0, None])
        def _observation(aidx: int, state: State) -> chex.Array:
            """Generate individual agent's observation"""

            ## TODO can we use the following pattern to do this better?
            # actions = jnp.array([actions[i] for i in self.agents])
            # aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            # action = actions.at[aidx].get()

            def get_player_obs(state: State) -> chex.Array:
                return jnp.array(
                    [
                        state.player.health_current,
                        state.player.health_max,
                        state.player.location_x,
                        state.player.location_y,
                        state.player.melee_attack_base_damage,
                        state.player.ranged_attack_base_damage,
                        state.player.melee_attack_range,
                        state.player.ranged_attack_range,
                        state.player.movement_points_current,
                        state.player.movement_points_max,
                        state.player.action_points_current,
                        state.player.action_points_max,
                        state.enemy.health_current,
                        state.enemy.health_max,
                        state.enemy.location_x,
                        state.enemy.location_y,
                        state.distance_to_enemy,
                        state.turn_count,
                        ]
                        )
            def get_enemy_obs(state: State) -> chex.Array:
                return jnp.array(
                    [
                        state.enemy.health_current,
                        state.enemy.health_max,
                        state.enemy.location_x,
                        state.enemy.location_y,
                        state.enemy.melee_attack_base_damage,
                        state.enemy.ranged_attack_base_damage,
                        state.enemy.melee_attack_range,
                        state.enemy.ranged_attack_range,
                        state.enemy.movement_points_current,
                        state.enemy.movement_points_max,
                        state.enemy.action_points_current,
                        state.enemy.action_points_max,
                        state.player.health_current,
                        state.player.health_max,
                        state.player.location_x,
                        state.player.location_y,
                        state.distance_to_enemy,
                        state.turn_count,
                        ]
                        )

            return jax.lax.cond(
                aidx == 0,
                lambda _: get_player_obs(state),
                lambda _: get_enemy_obs(state),
                operand=None
                )

        obs = _observation(self.agent_range, state)

        return {a: obs[i] for i, a in enumerate(self.agents)}
    
    def function_mapper(
            self,
            key: chex.PRNGKey,
            state: State,
            action: int,
            unit,
            target,
            ):
        # Dictionary mapping of input number to the corresponding function call
        action_fns = [action.execute for action in action_functions]
        # jax.debug.print("id {unit} do action: {action}", unit = unit.unit_id, action = action, ordered=True)

        new_state = lax.switch(action, action_fns, state, unit, target)
        new_state = new_state.replace(
            player = new_state.player.replace(
                available_actions = self.get_available_actions(new_state.player, new_state.enemy, new_state)
                ),
            enemy = new_state.enemy.replace(
                available_actions = self.get_available_actions(new_state.enemy, new_state.player, new_state)
                )
            )

        # Call the corresponding function based on the value of x
        return new_state

    def get_available_actions(
            self,
            unit,
            target,
            state: State) -> chex.Array:
        return jnp.array([action.is_valid(state, unit, target) for action in action_functions], dtype=jnp.int32)

    @partial(jax.jit, static_argnums=[0])

    def step_env(self, key: chex.PRNGKey, state: State, actions: Dict,
                 ) -> Tuple[chex.Array, State, Dict, Dict, Dict]:
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
        new_state = lax.cond(
                jnp.equal(aidx, 0),
                lambda _: self.function_mapper(key, old_state, action, old_state.player, old_state.enemy),
                lambda _: self.function_mapper(key, old_state, action, old_state.enemy, old_state.player),
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

    # def step_agent(self, key: chex.PRNGKey, state: State, aidx: int, action: int,
    #                ) -> Tuple[State, int]:
    #     """
    #     Execute the current player's action and its consequences
    #     """




    #     return state.replace(terminal=terminal,
    #                          last_moves=last_moves,
    #                          cur_player_idx=cur_player_idx,
    #                          out_of_lives=out_of_lives,
    #                          last_round_count=last_round_count,
    #                          bombed=bombed
    #                          ), reward

    def terminal(self, state: State) -> bool:
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