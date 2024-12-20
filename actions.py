# actions.py
import jax.numpy as jnp
from jax import lax, debug
import chex
from config import env_config
from data_classes import GameState
from utils import euclidean_distance, is_within_bounds, is_collision, do_invalid_move

class Action:
    """Base class for all actions in the game."""
    def __init__(self):
        self._ability_description = "Base action"
        self._base_cooldown = jnp.int32(0)
        self._parameter_1 = jnp.float32(0)
        self._parameter_2 = jnp.float32(0) 
        self._parameter_3 = jnp.float32(0)
    
    @property
    def ability_description(self) -> str:
        return self._ability_description
    
    @property
    def base_cooldown(self) -> chex.Array:
        return self._base_cooldown
        
    @property
    def parameter_1(self) -> chex.Array:
        return self._parameter_1
        
    @property
    def parameter_2(self) -> chex.Array:
        return self._parameter_2
        
    @property
    def parameter_3(self) -> chex.Array:
        return self._parameter_3
    
    def is_valid(
        self, 
        state: GameState, 
        source_id: int, 
        target_id: int,
        ability_slot: int = jnp.int32(-1),
    ) -> chex.Array:
        """Check if action is valid including cooldown check.
        
        Args:
            state: Current game state
            source_id: ID of unit attempting action
            target_id: ID of target unit
            ability_slot: Index of ability being used
            
        Returns:
            Boolean indicating if action can be performed
        """
        def check_ability_validity(_):
            # Check if ability is on cooldown
            current_cooldown = state.units.abilities[source_id, ability_slot, 2]
            not_on_cooldown = current_cooldown <= 0
            
            # Combine with specific ability validation
            return jnp.logical_and(
                not_on_cooldown,
                self._is_valid_ability(state, source_id, target_id, ability_slot)
            )
        
        def base_action_validity(_):
            return self._is_valid_ability(state, source_id, target_id, ability_slot)
            
        return lax.cond(
            ability_slot == -1,
            base_action_validity,
            check_ability_validity,
            operand=None
        )

    def _is_valid_ability(
        self,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int,
    ) -> chex.Array:
        """Override this in ability subclasses for specific validation."""
        return False

    def execute(
        self,
        key: chex.PRNGKey, 
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int = jnp.int32(-1),
    ) -> GameState:
        """Execute action if valid, handling cooldown for abilities.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            source_id: ID of unit performing action
            target_id: ID of target unit
            ability_slot: Index of ability being used (-1 for base actions)
            
        Returns:
            Updated game state
        """
        def do_action(_):
            # Execute ability logic
            new_state = self._perform_action(key, state, source_id, target_id, ability_slot)
            
            # Set cooldown if this is an ability action
            def set_cooldown(s):
                new_abilities = s.units.abilities.at[source_id, ability_slot, 2].set(
                    s.units.abilities[source_id, ability_slot, 1]  # Set current_cd to base_cd
                )
                new_units = s.units.replace(abilities=new_abilities)
                return s.replace(units=new_units)
            
            new_state = lax.cond(
                ability_slot >= 0,
                set_cooldown,
                lambda s: s,
                new_state
            )
            
            # Handle player state updates
            new_state = self._update_state_for_actor(new_state, source_id)
            return new_state

        def invalid_move(_):
            return do_invalid_move(state, source_id, target_id)

        return lax.cond(
            self.is_valid(state, source_id, target_id, ability_slot),
            do_action,
            invalid_move,
            None
        )

    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int = jnp.int32(-1)
    ) -> GameState:
        """Implement specific action logic.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            source_id: ID of unit performing action
            target_id: ID of target unit
            ability_slot: Index of ability being used
            
        Returns:
            Updated game state
        """
        raise NotImplementedError
        
    def _update_state_for_actor(
        self,
        state: GameState,
        source_id: int
    ) -> GameState:
        """Increment step counter if source unit is player controlled.
        
        Args:
            state: Current game state
            source_id: ID of unit that performed action
            
        Returns:
            Updated game state with incremented step counter if applicable
        """
        # Use HEROES_PER_TEAM to check if source is on player's team
        return lax.cond(
            source_id < env_config['HEROES_PER_TEAM'],
            lambda s: s.replace(steps=s.steps + 1),
            lambda s: s,
            state
        )