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
# from data_classes import UnitsState, EnvState, EnvParams
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

UnitsState = data_classes.UnitsState
GameState = data_classes.GameState

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

    @partial(jax.jit, static_argnums=[0])
    def get_legal_moves(self, state: GameState) -> Dict:
        """Get all agents' legal moves.
        
        Args:
            state: Current game state
            
        Returns:
            Dictionary mapping agent IDs to their legal moves
        """
        @partial(jax.vmap, in_axes=[0, None])
        def _legal_moves(aidx: int, state: GameState) -> chex.Array:
            """Get legal moves for a specific agent."""
            # Determine source unit index based on current player TODO rework when multiple units per team and have play order
            source_idx = jnp.where(
                aidx == 0,
                0,  # First team unit
                env_config['HEROES_PER_TEAM']  # Second team unit
            )
            # Return available actions for that unit
            return state.units.available_actions[source_idx]

        legal_moves = _legal_moves(self.agent_range, state)
        return {a: legal_moves[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, GameState]:
        """Reset the environment.
        
        Args:
            key: PRNG key for randomization
        
        Returns:
            Tuple containing:
            - Dict of initial observations for each agent
            - Initial GameState
        """
        key, key_pos, key_abilities = jax.random.split(key, 3)
        
        # Generate positions
        x1, y1, x2, y2 = generate_unique_pairs(key_pos)
        initial_distance = jnp.float32(euclidean_distance(x1, y1, x2, y2))

        # Sample ability indices for units and pool
        ability_idx1, ability_idx2, *pool_indices = jax.random.choice(
            key_abilities, num_abilities, shape=(2 + env_config['ABILITY_POOL_SIZE'],), replace=False
        )

        # Create abilities array initialization helper
        def create_initial_abilities(ability_indices: chex.Array, ability_params: chex.Array) -> chex.Array:
            """Creates initial abilities array with empty slots filled with defaults."""
            # Create empty ability array filled with defaults
            abilities = jnp.tile(
                jnp.array([-1, 0, 0, 0, 0, 0], dtype=jnp.float32),
                (env_config['HEROES_PER_TEAM'] * 2, env_config['ABILITIES_PER_HERO'], 1)
            )
            
            # Helper to get full ability parameters
            def get_full_params(idx):
                params = jax.lax.dynamic_slice(ability_params, (idx, 0), (1, 4))[0]
                return jnp.array([idx, params[0], 0, *params[1:]], dtype=jnp.float32)
            
            # Update first ability slot for each unit
            return abilities.at[:, 0].set(
                jax.vmap(get_full_params)(ability_indices)
            )

        # Create initial abilities for units
        abilities = create_initial_abilities(
            jnp.array([ability_idx1, ability_idx2]), 
            self.ability_params
        )

        # Create ability pool array
        def create_ability_pool(pool_indices: chex.Array, ability_params: chex.Array) -> chex.Array:
            """Creates ability pool array from indices."""
            def get_pool_ability(idx):
                params = jax.lax.dynamic_slice(ability_params, (idx, 0), (1, 4))[0]
                return jnp.array([idx, params[0], 0, *params[1:]], dtype=jnp.float32)
                
            return jax.vmap(get_pool_ability)(pool_indices)

        ability_pool = create_ability_pool(
            jnp.array(pool_indices), 
            self.ability_params
        )

        units = data_classes.create_unit_state(
            UnitsState,
            {
                'location': jnp.array([
                    [x1, y1],
                    [x2, y2]
                ], dtype=jnp.int32),
                'unit_id': jnp.arange(env_config['HEROES_PER_TEAM'] * 2, dtype=jnp.int32),
                'abilities': abilities
            }
        )

        # Create initial game state
        state = GameState(
            units=units,
            distance_to_enemy=initial_distance,
            steps=jnp.int32(0),
            turn_count=jnp.int32(0),
            previous_closest_distance=initial_distance,
            initial_distance=initial_distance,
            cur_player_idx=jnp.zeros(self.num_agents).at[0].set(1),
            terminal=False,
            # New ability pool handling
            ability_pool=ability_pool,
            ability_pool_picked=jnp.zeros(env_config['ABILITY_POOL_SIZE'], dtype=jnp.int32),
            pick_mode=jnp.int32(1),
            pick_count=jnp.int32(0)
        )

        # Update available actions for all units
        state = state.replace(
            units=state.units.replace(
                available_actions=self.get_available_actions(state)
            )
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
    
    def function_mapper(self, key: chex.PRNGKey, state: GameState, action: int, source_idx: int, target_idx: int) -> GameState:
        """Maps action indices to their corresponding functions.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            action: Action index to execute
            source_idx: Index of unit performing action
            target_idx: Index of target unit
            
        Returns:
            Updated game state
        """
        def do_base_action(key: chex.PRNGKey) -> GameState:
            """Execute a base action."""
            return lax.switch(
                action, 
                base_action_fns, 
                key, state, source_idx, target_idx, jnp.int32(-1)
            )
                
        def do_ability_action(key: chex.PRNGKey, slot_idx: int) -> GameState:
            """Execute an ability from the given slot."""
            # Get ability index from unit's abilities array
            ability_idx = jnp.int32(state.units.abilities[source_idx, slot_idx, 0])
            
            return lax.cond(
                ability_idx >= 0,
                lambda _: lax.switch(
                    ability_idx,
                    ability_action_fns, 
                    key, state, source_idx, target_idx, slot_idx
                ),
                lambda _: state,  # Return unchanged state if no ability in slot
                operand=None
            )
        
        key, key_ = jax.random.split(key)
        
        return lax.cond(
            action < num_base_actions,
            lambda _: do_base_action(key_),
            lambda _: lax.cond(
                action == num_base_actions,
                lambda _: do_ability_action(key_, jnp.int32(0)),  # First ability slot
                lambda _: do_ability_action(key_, jnp.int32(1)),  # Second ability slot
                operand=None
            ),
            operand=None
        )

    def get_available_actions(self, state: GameState) -> chex.Array:
        """Get available actions for all units.
        
        Args:
            state: Current game state
        
        Returns:
            Array of shape (HEROES_PER_TEAM * 2, num_actions) containing valid actions
        """
        def get_unit_actions(unit_idx: int) -> chex.Array:
            """Get available actions for a specific unit."""
            def check_base_action_validity(action_idx: int) -> chex.Array:
                """Check if a base action is valid for any target."""
                def check_target(target_idx: int) -> chex.Array:
                    is_self = target_idx == unit_idx
                    # Pass -1 as ability_slot for base actions
                    is_valid = base_action_functions[action_idx].is_valid(
                        state, unit_idx, target_idx, jnp.int32(-1)
                    )
                    return jnp.where(is_self, False, is_valid)
                
                target_validities = jax.vmap(check_target)(
                    jnp.arange(env_config['HEROES_PER_TEAM'] * 2)
                )
                return jnp.any(target_validities)
            
            def check_ability_validity(ability_idx: int, slot_idx: int) -> chex.Array:
                """Check if an ability is valid for any target."""
                def check_target(target_idx: int) -> chex.Array:
                    is_self = target_idx == unit_idx
                    is_valid = lax.cond(
                        ability_idx >= 0,
                        lambda _: lax.switch(
                            ability_idx,
                            [lambda: ability_action_functions[i].is_valid(
                                state, unit_idx, target_idx, slot_idx
                            ) for i in range(len(ability_action_functions))]
                        ),
                        lambda _: jnp.array(False),
                        operand=None
                    )
                    return jnp.where(is_self, False, is_valid)
                
                target_validities = jax.vmap(check_target)(
                    jnp.arange(env_config['HEROES_PER_TEAM'] * 2)
                )
                return jnp.any(target_validities)

            # Check base actions
            base_actions = jnp.array([
                check_base_action_validity(i) for i in range(len(base_action_functions))
            ], dtype=jnp.int32)
            
            # Get current unit's abilities
            unit_abilities = state.units.abilities[unit_idx]
            ability_1_idx = jnp.int32(unit_abilities[0, 0])  # First ability index
            ability_2_idx = jnp.int32(unit_abilities[1, 0])  # Second ability index
            
            # Check ability validities with their respective slots
            ability_1_valid = check_ability_validity(ability_1_idx, jnp.int32(0))
            ability_2_valid = check_ability_validity(ability_2_idx, jnp.int32(1))
            
            return jnp.concatenate([
                base_actions,
                jnp.array([ability_1_valid, ability_2_valid], dtype=jnp.int32)
            ])

        # Vectorize across all units
        return jax.vmap(get_unit_actions)(jnp.arange(env_config['HEROES_PER_TEAM'] * 2))

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self,
        key: chex.PRNGKey,
        state: GameState,
        actions: Dict
    ) -> Tuple[Dict, GameState, Dict, Dict, Dict]:
        """Steps the environment forward by one timestep.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            actions: Dictionary mapping agent IDs to their chosen actions
            
        Returns:
            Tuple containing:
            - Dict of observations for each agent
            - Updated game state
            - Dict of rewards for each agent
            - Dict of done flags
            - Dict of debug info
        """
        # Convert actions dict to array
        actions = jnp.array([actions[i] for i in self.agents])
        
        # Get current player's index and action
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        action = actions.at[aidx].get()[0]
        
        # Store old state for reward calculation
        old_state = state
        
        # Determine source unit index based on current player
        source_idx = jnp.where(
            aidx == 0,
            0,  # First team unit
            env_config['HEROES_PER_TEAM']  # Second team unit
        )
        
        # Perform action using function mapper
        key, key_ = jax.random.split(key)
        new_state = self.function_mapper(
            key_,
            state,
            action,
            source_idx,
            jnp.where(source_idx == 0, env_config['HEROES_PER_TEAM'], 0) # TODO: Make dynamic
        )
        
        # Update available actions
        new_state = new_state.replace(
            units=new_state.units.replace(
                available_actions=self.get_available_actions(new_state)
            )
        )
        
        # Calculate rewards
        player_reward, enemy_reward = reward_function(old_state, new_state, action)
        
        # Check termination conditions
        team_1_health = jnp.sum(new_state.units.health[:env_config['HEROES_PER_TEAM'], 0])
        team_2_health = jnp.sum(new_state.units.health[env_config['HEROES_PER_TEAM']:, 0])
        
        any_team_dead = jnp.logical_or(team_1_health <= 0, team_2_health <= 0)
        steps_exceeded = new_state.steps >= env_config['MAX_STEPS']
        done = jnp.logical_or(any_team_dead, steps_exceeded)
        
        # Update terminal state
        new_state = new_state.replace(terminal=done)
        
        # Package return values
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        
        rewards = {
            self.agents[0]: player_reward,
            self.agents[1]: enemy_reward,
        }
        rewards["__all__"] = 0
        
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