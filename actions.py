# actions.py
import jax.numpy as jnp
from jax import lax, debug
import chex
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
    def ability_description(self):
        return self._ability_description
    
    @property
    def base_cooldown(self):
        return self._base_cooldown
        
    @property
    def parameter_1(self):
        return self._parameter_1
        
    @property
    def parameter_2(self):
        return self._parameter_2
        
    @property
    def parameter_3(self):
        return self._parameter_3
    
    def is_valid(self, state, unit, target):
        """Default validity check."""
        return False

    def execute(self, key: chex.PRNGKey, state, unit, target, ability_idx=jnp.int32(-1)):
        """Template method pattern for executing actions."""
        def do_action(_):
            new_state = self._perform_action(key, state, unit, target, ability_idx)
            updated_unit = lax.cond(
                jnp.equal(state.player.unit_id, unit.unit_id),
                lambda _: new_state.player,
                lambda _: new_state.enemy,
                operand=None
            )

            new_unit = self.update_ability_cooldown(updated_unit, ability_idx)

            final_state = lax.cond(
                jnp.equal(state.player.unit_id, unit.unit_id),
                lambda: new_state.replace(player=new_unit),
                lambda: new_state.replace(enemy=new_unit)
            )

            return self._update_state_for_actor(final_state, unit)

        def invalid_move(_):
            return do_invalid_move(state, unit, target)

        return lax.cond(
            self.is_valid(state, unit, target, ability_idx),
            do_action,
            invalid_move,
            None,
        )

    def _perform_action(self, key: chex.PRNGKey, state, unit, target, ability_idx=jnp.int32(-1)):
        """Override this to implement specific action logic."""
        raise NotImplementedError
        
    def _update_state_for_actor(self, state, unit):# TODO: check if this is correct
        """Updates step counter if player acted."""
        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            lambda s: s.replace(steps=jnp.int32(s.steps + 1)),
            lambda s: s,
            state
        )

    def update_ability_cooldown(self, unit, ability_idx):
        """Updates cooldown for the specified ability slot"""
        def set_cooldown(_):
            return lax.switch(ability_idx,
                [
                    lambda op: unit.replace(
                        ability_state_1=unit.ability_state_1.replace(
                            current_cooldown=jnp.int32(unit.ability_state_1.base_cooldown)
                        )
                    ),
                    # Add more slots as needed
                ],
                None  # Pass operand to switch
            )
        
        return lax.cond(
            jnp.greater_equal(ability_idx, 0),  # Check if it's a valid ability index
            set_cooldown,
            lambda _: unit,  # No cooldown update for non-ability actions
            None
        )