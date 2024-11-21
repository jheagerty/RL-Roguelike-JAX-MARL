# base_actions.py
import jax.numpy as jnp
from jax import lax, debug
import chex
from utils import euclidean_distance, is_within_bounds, is_collision, do_invalid_move, do_attack
from actions import Action
from data_classes import AttackType, DamageType


base_action_registry = {}

def register_action(name, create_fn):
    base_action_registry[name] = create_fn

# Example registration for MoveAction
register_action("MoveAction", lambda: [MoveAction(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx != 0 or dy != 0])

# Register other actions
register_action("MeleeAttackAction", lambda: [MeleeAttackAction()])
register_action("RangedAttackAction", lambda: [RangedAttackAction()])
register_action("EndTurnAction", lambda: [EndTurnAction()])

class MoveAction(Action):
    def __init__(self, dx, dy):
        super().__init__()
        self.dx = dx
        self.dy = dy
        
        # Create direction description
        direction = ""
        if dy > 0: direction += "up"
        if dy < 0: direction += "down"
        if dx > 0: direction += "right"
        if dx < 0: direction += "left"
        
        movement_type = "diagonal" if (dx != 0 and dy != 0) else "orthogonal"
        distance = jnp.sqrt(dx**2 + dy**2)
        
        self._ability_description = f"Move {direction} ({movement_type}, {distance:.1f} movement points)"

    def is_valid(self, state, unit, target, ability_idx):
        new_x, new_y = unit.location_x + self.dx, unit.location_y + self.dy
        within_bounds = is_within_bounds(new_x, new_y)
        collision_enemy = is_collision(new_x, new_y, state.enemy.location_x, state.enemy.location_y)
        collision_player = is_collision(new_x, new_y, state.player.location_x, state.player.location_y)
        collision = jnp.logical_or(collision_enemy, collision_player)
        valid_move = jnp.logical_and(within_bounds, jnp.logical_not(collision))
        movement_cost = jnp.float32(jnp.sqrt(self.dx**2 + self.dy**2))
        enough_movement_points = unit.movement_points_current >= movement_cost
        return jnp.logical_and(valid_move, enough_movement_points)

    def _perform_action(self, key: chex.PRNGKey, state, unit, target, ability_idx=jnp.int32(-1)):
        new_x, new_y = unit.location_x + self.dx, unit.location_y + self.dy
        distance = jnp.float32(jnp.sqrt(self.dx**2 + self.dy**2))
        unit_movement_points = jnp.float32(unit.movement_points_current - distance)
        
        distance_to_enemy = lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            lambda: jnp.float32(euclidean_distance(new_x, new_y, state.enemy.location_x, state.enemy.location_y)),
            lambda: jnp.float32(euclidean_distance(state.player.location_x, state.player.location_y, new_x, new_y)),
        )
        
        new_unit = unit.replace(
            location_x=new_x,
            location_y=new_y,
            movement_points_current=unit_movement_points,
        )
        
        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            lambda: state.replace(
                player=new_unit,
                distance_to_enemy=distance_to_enemy,
                previous_closest_distance=jnp.minimum(state.previous_closest_distance, distance_to_enemy),
            ),
            lambda: state.replace(
                enemy=new_unit,
                distance_to_enemy=distance_to_enemy,
                previous_closest_distance=jnp.minimum(state.previous_closest_distance, distance_to_enemy),
            ),
        )

class MeleeAttackAction(Action):#TODO: update to default physical but can be other types
    def __init__(self):
        super().__init__()

    def is_valid(self, state, unit, target, ability_idx):
        enough_action_points = unit.action_points_current >= 1
        within_range = state.distance_to_enemy <= unit.melee_attack_range
        return jnp.logical_and(enough_action_points, within_range)

    def _perform_action(self, key: chex.PRNGKey, state, unit, target, ability_idx=jnp.int32(-1)):
        new_unit = unit.replace(
            action_points_current=jnp.float32(unit.action_points_current - 1),
            base_melee_attack_count = unit.base_melee_attack_count + 1,
        )
        new_unit, new_target = do_attack(new_unit, target, 
                                       AttackType.MELEE, 
                                       DamageType.PHYSICAL)
        
        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            lambda: state.replace(player=new_unit, enemy=new_target),
            lambda: state.replace(player=new_target, enemy=new_unit),
        )

class RangedAttackAction(Action):
    def __init__(self):
        super().__init__()

    def is_valid(self, state, unit, target, ability_idx):
        enough_action_points = unit.action_points_current >= 1
        within_range = state.distance_to_enemy <= unit.ranged_attack_range
        return jnp.logical_and(enough_action_points, within_range)

    def _perform_action(self, key: chex.PRNGKey, state, unit, target, ability_idx=jnp.int32(-1)):
        new_unit = unit.replace(
            action_points_current=jnp.float32(unit.action_points_current - 1),
            base_ranged_attack_count = unit.base_ranged_attack_count + 1,
        )
        new_unit, new_target = do_attack(new_unit, target,
                                       AttackType.RANGED,
                                       DamageType.PHYSICAL)
        
        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            lambda: state.replace(player=new_unit, enemy=new_target),
            lambda: state.replace(player=new_target, enemy=new_unit),
        )

class EndTurnAction(Action):
    def __init__(self):
        super().__init__()
        self.num_agents = 2

    def is_valid(self, state, unit, target, ability_idx):
        return True

    def _perform_action(self, key: chex.PRNGKey, state, unit, target, ability_idx=jnp.int32(-1)):
        def reduce_cooldowns(unit):
            def update_ability_state_cooldown(ability_state):
                return ability_state.replace(
                    current_cooldown=jnp.maximum(0, ability_state.current_cooldown - 1)
                )
            
            # Use lax.switch for ability states
            return lax.switch(0,  # Currently only handling first ability
                [
                    lambda _: unit.replace(
                        ability_state_1=update_ability_state_cooldown(unit.ability_state_1)
                    ),
                    # Add more cases here for additional abilities
                ],
                None
            )

        def end_turn_player():
            new_player_state = reduce_cooldowns(state.player).replace(
                action_points_current=jnp.float32(state.player.action_points_max),
                movement_points_current=jnp.float32(state.player.movement_points_max),
            )
            aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            aidx = (aidx + 1) % self.num_agents
            cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)
            return state.replace(
                player=new_player_state,
                turn_count=jnp.int32(state.turn_count + 1),
                cur_player_idx=cur_player_idx,
            )

        def end_turn_enemy():
            new_enemy_state = reduce_cooldowns(state.enemy).replace(
                action_points_current=jnp.float32(state.enemy.action_points_max),
                movement_points_current=jnp.float32(state.enemy.movement_points_max),
            )
            aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            aidx = (aidx + 1) % self.num_agents
            cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)
            return state.replace(
                enemy=new_enemy_state,
                cur_player_idx=cur_player_idx,
            )

        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            end_turn_player,
            end_turn_enemy,
        )