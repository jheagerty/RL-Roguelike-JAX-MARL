import jax.numpy as jnp
from jax import lax, debug
from utils import euclidean_distance, is_within_bounds, is_collision, do_invalid_move

action_registry = {}

def register_action(name, create_fn):
    def register_action(name: str, create_fn: Callable) -> None:
        """
        Registers a new action in the action registry.
        This function adds a new action to the global action registry dictionary,
        allowing the action to be referenced and created later by name.
        Args:
            name (str): The unique identifier/name for the action being registered
            create_fn (Callable): A factory function that creates an instance of the action
                                The function should take no arguments and return a list of action instances
        Returns:
        Example:
            >>> register_action("MoveAction", lambda: [MoveAction()])
        Note:
            - Action names should be unique across the registry
            - The create_fn should return a list of action instances, even for single actions
            - This function modifies the global action_registry dictionary
        """
    action_registry[name] = create_fn

# suicide:
# Register suicide action
register_action("SuicideAction", lambda: [SuicideAction()])

class SuicideAction:
    """Represents a suicide attack action where both attacker and target take damage.
    This action deals damage to both the attacker and the target when executed within range.
    This class implements a melee-range attack that requires action points and damages both units.
    Attributes:
        damage (float): Amount of damage dealt to both units (default: 5.0)
    Methods:
        is_valid(state, unit, target): Checks if the action can be performed
        execute(state, unit, target): Performs the suicide attack
    Args for methods:
        state: Current game state containing unit positions and attributes 
        unit: The attacking unit attempting the suicide action
        target: The target unit to attack
    Returns:
        execute(): Updated game state after the action is performed
        is_valid(): Boolean indicating if action can be performed
    Notes:
        - Requires 1 action point
        - Range check of 1.5 units or less
        - Both units take equal damage
        - Cannot reduce health below 0
        - Updates step counter when player performs action
    """
    def __init__(self):
        self.damage = 5.0

    def is_valid(self, state, unit, target):
        enough_action_points = unit.action_points_current >= 1
        within_range = state.distance_to_enemy <= 1.5
        return jnp.logical_and(enough_action_points, within_range)

    def execute(self, state, unit, target):
        def do_suicide(_):
            target_health = jnp.maximum(0, target.health_current - self.damage)
            unit_health = jnp.maximum(0, unit.health_current - self.damage)
            unit_action_points = unit.action_points_current - 1
            
            new_target = target.replace(health_current=target_health)
            new_unit = unit.replace(
                health_current=unit_health,
                action_points_current=unit_action_points
            )
            
            def suicide_as_player():
                return state.replace(
                    player=new_unit,
                    enemy=new_target,
                    steps=state.steps + 1
                )
                
            def suicide_as_enemy():
                return state.replace(
                    player=new_target,
                    enemy=new_unit
                )

            return lax.cond(
                jnp.equal(state.player.unit_id, unit.unit_id),
                suicide_as_player,
                suicide_as_enemy
            )

        def invalid_move(_):
            return do_invalid_move(state, unit, target)

        return lax.cond(
            self.is_valid(state, unit, target),
            do_suicide,
            invalid_move,
            None
        )

# regen:

# stun_nuke:

# minus_armour:

# wound