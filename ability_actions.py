import jax.numpy as jnp
from jax import lax, debug
from utils import euclidean_distance, is_within_bounds, is_collision, do_invalid_move
from actions import Action

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

class SuicideAction(Action):
    def __init__(self):
        super().__init__()
        self.damage = 5.0
        self.range = 8.0
        
    def is_valid(self, state, unit, target):
        enough_action_points = unit.action_points_current >= 1
        within_range = state.distance_to_enemy <= self.range
        return jnp.logical_and(enough_action_points, within_range)

    def _perform_action(self, state, unit, target):
        # Generate all 8 adjacent grid positions
        adjacent_positions = [
            (target.location_x + dx, target.location_y + dy)
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        ]
        
        # Find closest valid adjacent position
        # Start with first position
        best_x, best_y = adjacent_positions[0]
        best_dist = euclidean_distance(unit.location_x, unit.location_y, best_x, best_y)
        
        # Compare with other positions using lax.fori_loop
        def update_best_position(i, val):
            x, y = adjacent_positions[i]
            dist = euclidean_distance(unit.location_x, unit.location_y, x, y)
            use_new = jnp.logical_and(
                is_within_bounds(x, y),
                dist < val[2]
            )
            return lax.cond(
                use_new,
                lambda _: (x, y, dist),
                lambda _: val,
                None
            )
            
        new_x, new_y, _ = lax.fori_loop(1, 8, update_best_position, (best_x, best_y, best_dist))

        # Apply damage
        target_health = jnp.maximum(0, target.health_current - self.damage)
        unit_health = jnp.maximum(0, unit.health_current - self.damage)
        unit_action_points = unit.action_points_current - 1
        
        # Update units with grid-aligned position
        new_target = target.replace(health_current=jnp.float32(target_health))
        new_unit = unit.replace(
            health_current=jnp.float32(unit_health),
            action_points_current=jnp.float32(unit_action_points),
            location_x=jnp.float32(new_x),
            location_y=jnp.float32(new_y)
        )
        
        # Calculate distance after teleport
        new_distance = euclidean_distance(new_x, new_y, target.location_x, target.location_y)
        
        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            lambda: state.replace(
                player=new_unit, 
                enemy=new_target,
                distance_to_enemy=new_distance
            ),
            lambda: state.replace(
                player=new_target, 
                enemy=new_unit,
                distance_to_enemy=new_distance
            )
        )

# Steal Strength - reduce enemy strength and increase own strength
# Strength Regen - regen health based on your strength
# Add barrier - add a barrier based on resolve
# Mana Burn
# Multi Attack
# Return
# Fury Swipes
# Push
# Frost Arrows
# Stun
# Hook
# Lifesteal / feast
# Int steal
# Int based nuke
# Armour reduction
# Add barrier
# Spellsteal
# Fracture casting