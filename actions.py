# actions.py
import jax.numpy as jnp
from jax import lax, debug
from utils import euclidean_distance, is_within_bounds, is_collision, do_invalid_move

class Action:
    """Base class for all actions in the game."""
    def __init__(self):
        self._ability_description = "Base action"
    
    @property
    def ability_description(self):
        """Returns description of the action."""
        return self._ability_description
    
    def is_valid(self, state, unit, target):
        """Default validity check."""
        return True

    def execute(self, state, unit, target):
        """Template method pattern for executing actions."""
        def do_action(_):
            new_state = self._perform_action(state, unit, target)
            return self._update_state_for_actor(new_state, unit)

        def invalid_move(_):
            return do_invalid_move(state, unit, target)

        return lax.cond(
            self.is_valid(state, unit, target),
            do_action,
            invalid_move,
            None,
        )

    def _perform_action(self, state, unit, target):
        """Override this to implement specific action logic."""
        raise NotImplementedError
        
    def _update_state_for_actor(self, state, unit):
        """Updates step counter if player acted."""
        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            lambda s: s.replace(steps=jnp.int32(s.steps + 1)),
            lambda s: s,
            state
        )

