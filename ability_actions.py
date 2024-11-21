import jax.numpy as jnp
import chex
from jax import lax, debug
from utils import euclidean_distance, is_within_bounds, is_collision, do_invalid_move, do_damage
from actions import Action
from data_classes import DamageType

# Create global registry
ability_registry = {}

def register_action(name, create_fn):
    ability_registry[name] = create_fn

# suicide:
# Register suicide action
register_action("SuicideAction", lambda: [SuicideAction()])

class SuicideAction(Action):
    def __init__(self):
        super().__init__()
        # Define all parameters as class variables during initialization
        self._ability_description = "Deal damage based on strength to an enemy and take damage yourself"
        self._base_cooldown = jnp.int32(3)
        self._parameter_1 = jnp.float32(8)  # range
        self._parameter_2 = jnp.float32(5)  # base_damage
        
    def is_valid(self, state, unit, target):
        enough_action_points = unit.action_points_current >= 1
        within_range = state.distance_to_enemy <= self.parameter_1
        return jnp.logical_and(enough_action_points, within_range)

    def _perform_action(self, key: chex.PRNGKey, state, unit, target, ability_idx=jnp.int32(-1)):
        # Create position offsets as a static array
        position_offsets = jnp.array([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ])
        
        # Initialize best position
        init_pos = position_offsets[0]
        init_x = target.location_x + init_pos[0]
        init_y = target.location_y + init_pos[1]
        init_dist = euclidean_distance(unit.location_x, unit.location_y, init_x, init_y)
        
        def update_best_position(i, val):
            offset = lax.dynamic_slice(position_offsets, (i, 0), (1, 2))[0]
            x = target.location_x + offset[0]
            y = target.location_y + offset[1]
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
        
        new_x, new_y, _ = lax.fori_loop(1, 8, update_best_position, (init_x, init_y, init_dist))
        
        # Rest of the function remains the same...
        damage_dealt = self.parameter_2 + unit.strength_current
        new_unit, new_target = do_damage(unit, target, damage_dealt, DamageType.PURE)
        new_unit, _ = do_damage(unit, new_unit, self.parameter_2, DamageType.PURE)
        
        new_unit = new_unit.replace(
            action_points_current=new_unit.action_points_current - 1,
            location_x=jnp.int32(new_x),
            location_y=jnp.int32(new_y),
            suicide_ability_count = unit.suicide_ability_count + 1,
        )
        
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
# Register the new action
register_action("StealStrengthAction", lambda: [StealStrengthAction()])

class StealStrengthAction(Action):
    def __init__(self):
        super().__init__()
        self._ability_description = "Steal 2 strength from the target"
        self._base_cooldown = jnp.int32(1)
        self._parameter_1 = jnp.float32(4)  # range
        self._parameter_2 = jnp.float32(2)  # strength_steal_amount

    def is_valid(self, state, unit, target):
        enough_action_points = unit.action_points_current >= 1
        within_range = state.distance_to_enemy <= self.parameter_1
        return jnp.logical_and(enough_action_points, within_range)

    def _perform_action(self, key: chex.PRNGKey, state, unit, target, ability_idx=jnp.int32(-1)):
        # Reduce target's strength
        new_target = target.replace(
            strength_current=jnp.maximum(0, target.strength_current - self.parameter_2)
        )
        
        # Increase caster's strength and reduce action points
        new_unit = unit.replace(
            strength_current=unit.strength_current + self.parameter_2,
            action_points_current=unit.action_points_current - 1,
            steal_strength_ability_count = unit.steal_strength_ability_count + 1,
        )

        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            lambda: state.replace(player=new_unit, enemy=new_target),
            lambda: state.replace(player=new_target, enemy=new_unit)
        )
    
# Multi Attack
# Frost Arrows
# Strength Regen - regen health based on your strength
# Add barrier - add a barrier based on resolve
# Mana Burn
# Return
# Fury Swipes
# Push
# Stun
# Hook
# Lifesteal / feast
# Int steal
# Int based nuke
# Armour reduction
# Add barrier
# Spellsteal
# Fracture casting