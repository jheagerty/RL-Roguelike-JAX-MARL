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
register_action("PickPoolAbility1", lambda: [PickPoolAbility1()])
register_action("PickPoolAbility2", lambda: [PickPoolAbility2()])
register_action("PickPoolAbility3", lambda: [PickPoolAbility3()])

class PickPoolAbility1(Action):
    def __init__(self):
        super().__init__()
        self.num_agents = 2

    def is_valid(self, state, unit, target):
        return jnp.logical_and(
            state.pick_mode == 1,
            state.pool_ability_1_picked == 0
        )
    def _perform_action(self, key, state, unit, target, ability_idx=jnp.int32(-1)):
        # Update ability state 2 field by field
        new_ability_state_2 = unit.ability_state_2.replace(
            ability_index=jnp.int32(state.pool_ability_1.ability_index),
            base_cooldown=jnp.int32(state.pool_ability_1.base_cooldown),
            current_cooldown=jnp.int32(state.pool_ability_1.current_cooldown),
            parameter_1=jnp.int32(state.pool_ability_1.parameter_1),
            parameter_2=jnp.int32(state.pool_ability_1.parameter_2),
            parameter_3=jnp.int32(state.pool_ability_1.parameter_3)
        )

        # Update unit with new ability
        updated_unit = unit.replace(
            ability_state_2=new_ability_state_2,
            action_points_current=jnp.float32(0)  # End turn by setting AP to 0
        )

        # Calculate new pick count and pick mode
        new_pick_count = state.pick_count + 1
        new_pick_mode = jnp.where(new_pick_count >= 2, 0, 1)
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        aidx = (aidx + 1) % self.num_agents
        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)

        def update_player():
            return state.replace(
                pool_ability_1_picked=jnp.int32(1),
                player=updated_unit,
                enemy=state.enemy,
                pick_count=new_pick_count,
                pick_mode=new_pick_mode,
                cur_player_idx=cur_player_idx  # Switch active player
            )
        
        def update_enemy():
            return state.replace(
                pool_ability_1_picked=jnp.int32(1),
                player=state.player,
                enemy=updated_unit,
                pick_count=new_pick_count,
                pick_mode=new_pick_mode,
                cur_player_idx=cur_player_idx  # Switch active player
            )
            
        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            update_player,
            update_enemy
        )

class PickPoolAbility2(Action):
    def __init__(self):
        super().__init__()
        self.num_agents = 2

    def is_valid(self, state, unit, target):
        return jnp.logical_and(
            state.pick_mode == 1,
            state.pool_ability_2_picked == 0
        )

    def _perform_action(self, key, state, unit, target, ability_idx=jnp.int32(-1)):
        # Update ability state 2 field by field
        new_ability_state_2 = unit.ability_state_2.replace(
            ability_index=jnp.int32(state.pool_ability_2.ability_index),
            base_cooldown=jnp.int32(state.pool_ability_2.base_cooldown),
            current_cooldown=jnp.int32(state.pool_ability_2.current_cooldown),
            parameter_1=jnp.int32(state.pool_ability_2.parameter_1),
            parameter_2=jnp.int32(state.pool_ability_2.parameter_2),
            parameter_3=jnp.int32(state.pool_ability_2.parameter_3)
        )

        # Update unit with new ability and end turn
        updated_unit = unit.replace(
            ability_state_2=new_ability_state_2,
            action_points_current=jnp.float32(0)  # End turn by setting AP to 0
        )

        # Calculate new pick count and pick mode
        new_pick_count = state.pick_count + 1
        new_pick_mode = jnp.where(new_pick_count >= 2, 0, 1)
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        aidx = (aidx + 1) % self.num_agents
        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)

        def update_player():
            return state.replace(
                pool_ability_2_picked=jnp.int32(1),
                player=updated_unit,
                enemy=state.enemy,
                pick_count=new_pick_count,
                pick_mode=new_pick_mode,
                cur_player_idx=cur_player_idx  # Switch active player
            )
            
        def update_enemy():
            return state.replace(
                pool_ability_2_picked=jnp.int32(1),
                player=state.player,
                enemy=updated_unit,
                pick_count=new_pick_count,
                pick_mode=new_pick_mode,
                cur_player_idx=state.cur_player_idx  # Switch active player
            )
            
        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            update_player,
            update_enemy
        )

class PickPoolAbility3(Action):
    def __init__(self):
        super().__init__()
        self.num_agents = 2

    def is_valid(self, state, unit, target):
        return jnp.logical_and(
            state.pick_mode == 1,
            state.pool_ability_3_picked == 0
        )

    def _perform_action(self, key, state, unit, target, ability_idx=jnp.int32(-1)):
        # Update ability state 2 field by field
        new_ability_state_2 = unit.ability_state_2.replace(
            ability_index=jnp.int32(state.pool_ability_3.ability_index),
            base_cooldown=jnp.int32(state.pool_ability_3.base_cooldown),
            current_cooldown=jnp.int32(state.pool_ability_3.current_cooldown),
            parameter_1=jnp.int32(state.pool_ability_3.parameter_1),
            parameter_2=jnp.int32(state.pool_ability_3.parameter_2),
            parameter_3=jnp.int32(state.pool_ability_3.parameter_3)
        )

        # Update unit with new ability and end turn
        updated_unit = unit.replace(
            ability_state_2=new_ability_state_2,
            action_points_current=jnp.float32(0)  # End turn by setting AP to 0
        )

        # Calculate new pick count and pick mode
        new_pick_count = state.pick_count + 1
        new_pick_mode = jnp.where(new_pick_count >= 2, 0, 1)
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        aidx = (aidx + 1) % self.num_agents
        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)

        def update_player():
            return state.replace(
                pool_ability_3_picked=jnp.int32(1),
                player=updated_unit,
                enemy=state.enemy,
                pick_count=new_pick_count,
                pick_mode=new_pick_mode,
                cur_player_idx=cur_player_idx  # Switch active player
            )
            
        def update_enemy():
            return state.replace(
                pool_ability_3_picked=jnp.int32(1),
                player=state.player,
                enemy=updated_unit,
                pick_count=new_pick_count,
                pick_mode=new_pick_mode,
                cur_player_idx=cur_player_idx  # Switch active player
            )
            
        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            update_player,
            update_enemy
        )
    
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

    def is_valid(self, state, unit, target):
        new_x, new_y = unit.location_x + self.dx, unit.location_y + self.dy
        within_bounds = is_within_bounds(new_x, new_y)
        collision_enemy = is_collision(new_x, new_y, state.enemy.location_x, state.enemy.location_y)
        collision_player = is_collision(new_x, new_y, state.player.location_x, state.player.location_y)
        collision = jnp.logical_or(collision_enemy, collision_player)
        valid_move = jnp.logical_and(within_bounds, jnp.logical_not(collision))
        movement_cost = jnp.float32(jnp.sqrt(self.dx**2 + self.dy**2))
        enough_movement_points = unit.movement_points_current >= movement_cost
        not_pick_mode = state.pick_mode == 0
        return jnp.logical_and(jnp.logical_and(valid_move, enough_movement_points), not_pick_mode)

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

    def is_valid(self, state, unit, target):
        enough_action_points = unit.action_points_current >= 1
        within_range = state.distance_to_enemy <= unit.melee_attack_range
        not_pick_mode = state.pick_mode == 0
        return jnp.logical_and(jnp.logical_and(enough_action_points, within_range), not_pick_mode)

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

    def is_valid(self, state, unit, target):
        enough_action_points = unit.action_points_current >= 1
        within_range = state.distance_to_enemy <= unit.ranged_attack_range
        not_pick_mode = state.pick_mode == 0
        return jnp.logical_and(jnp.logical_and(enough_action_points, within_range), not_pick_mode)

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

    def is_valid(self, state, unit, target):
        not_pick_mode = state.pick_mode == 0
        return not_pick_mode

    def _perform_action(self, key: chex.PRNGKey, state, unit, target, ability_idx=jnp.int32(-1)):
        def end_turn_player():
            new_player_state = state.player.replace(
                action_points_current=jnp.float32(state.player.action_points_max),
                movement_points_current=jnp.float32(state.player.movement_points_max),
                end_turn_count = unit.end_turn_count + 1,
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
            new_enemy_state = state.enemy.replace(
                action_points_current=jnp.float32(state.enemy.action_points_max),
                movement_points_current=jnp.float32(state.enemy.movement_points_max),
                end_turn_count = unit.end_turn_count + 1,
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