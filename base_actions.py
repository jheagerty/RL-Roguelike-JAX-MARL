# base_actions.py
import jax.numpy as jnp
from jax import lax, debug
from utils import euclidean_distance, is_within_bounds, is_collision, do_invalid_move

action_registry = {}

def register_action(name, create_fn):
    action_registry[name] = create_fn

# Example registration for MoveAction
register_action("MoveAction", lambda: [MoveAction(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx != 0 or dy != 0])

# Register other actions
register_action("MeleeAttackAction", lambda: [MeleeAttackAction()])
register_action("RangedAttackAction", lambda: [RangedAttackAction()])
register_action("EndTurnAction", lambda: [EndTurnAction()])

class MoveAction:
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def is_valid(self, state, unit, target):
        new_x, new_y = unit.location_x + self.dx, unit.location_y + self.dy
        within_bounds = is_within_bounds(new_x, new_y)
        collision_enemy = is_collision(new_x, new_y, state.enemy.location_x, state.enemy.location_y)
        collision_player = is_collision(new_x, new_y, state.player.location_x, state.player.location_y)
        collision = jnp.logical_or(collision_enemy, collision_player)
        valid_move = jnp.logical_and(within_bounds, jnp.logical_not(collision))
        # Convert movement points check to float32
        movement_cost = jnp.float32(jnp.sqrt(self.dx**2 + self.dy**2))
        enough_movement_points = unit.movement_points_current >= movement_cost
        return jnp.logical_and(valid_move, enough_movement_points)

    def execute(self, state, unit, target):
        def do_move(_):
            new_x, new_y = unit.location_x + self.dx, unit.location_y + self.dy
            # Ensure consistent float32 type for movement calculations
            distance = jnp.float32(jnp.sqrt(self.dx**2 + self.dy**2))
            unit_movement_points = jnp.float32(unit.movement_points_current - distance)
            
            def distance_to_enemy_from_player():
                return jnp.float32(euclidean_distance(new_x, new_y, state.enemy.location_x, state.enemy.location_y))
            def distance_to_player_from_enemy():
                return jnp.float32(euclidean_distance(state.player.location_x, state.player.location_y, new_x, new_y))
                
            distance_to_enemy = lax.cond(
                jnp.equal(state.player.unit_id, unit.unit_id),
                distance_to_enemy_from_player,
                distance_to_player_from_enemy,
            )
            
            new_unit = unit.replace(
                location_x=new_x,
                location_y=new_y,
                movement_points_current=unit_movement_points,  # Now consistently float32
            )
            
            def move_player():
                return state.replace(
                    player=new_unit,
                    distance_to_enemy=distance_to_enemy,
                    steps=jnp.int32(state.steps + 1),
                    previous_closest_distance=jnp.float32(jnp.minimum(state.previous_closest_distance, distance_to_enemy)),
                )
            def move_enemy():
                return state.replace(
                    enemy=new_unit,
                    distance_to_enemy=distance_to_enemy,
                    previous_closest_distance=jnp.float32(jnp.minimum(state.previous_closest_distance, distance_to_enemy)),
                )
                
            new_state = lax.cond(
                jnp.equal(state.player.unit_id, unit.unit_id),
                move_player,
                move_enemy,
            )

            return new_state

        def invalid_move(_):
            return do_invalid_move(state, unit, target)

        result_state = lax.cond(
            self.is_valid(state, unit, target),
            do_move,
            invalid_move,
            None,
        )

        return result_state

class MeleeAttackAction:
    def __init__(self):
        pass

    def is_valid(self, state, unit, target):
        enough_action_points = unit.action_points_current >= 1
        within_range = state.distance_to_enemy <= unit.melee_attack_range
        return jnp.logical_and(enough_action_points, within_range)

    def execute(self, state, unit, target):
        def do_melee_attack(_):
            target_health_after = jnp.maximum(0, target.health_current - unit.melee_base_attack_damage)
            unit_action_points = unit.action_points_current - 1
            new_target = target.replace(health_current=jnp.float32(target_health_after))
            new_unit = unit.replace(action_points_current=jnp.float32(unit_action_points))
            
            def do_melee_attack_player():
                return state.replace(
                player=new_unit,
                enemy=new_target,
                steps=jnp.int32(state.steps + 1),
            )
            def do_melee_attack_enemy():
                return state.replace(
                player=new_target,
                enemy=new_unit,
            )

            new_state = lax.cond(
                jnp.equal(state.player.unit_id, unit.unit_id),
                do_melee_attack_player,
                do_melee_attack_enemy,
            )

            return new_state

        def invalid_move(_):
            # debug.print("invalid move: melee attack, unit: {unit}", unit = unit.unit_id)
            return do_invalid_move(state, unit, target)

        return lax.cond( # TODO USE LAX SWITCH ONCE MORE UNITS
            self.is_valid(state, unit, target),
            do_melee_attack,
            invalid_move,
            None,
        )

class RangedAttackAction:
    def __init__(self):
        pass

    def is_valid(self, state, unit, target):
        enough_action_points = unit.action_points_current >= 1
        within_range = state.distance_to_enemy <= unit.ranged_attack_range
        return jnp.logical_and(enough_action_points, within_range)

    def execute(self, state, unit, target):
        def do_ranged_attack(_):
            target_health_after = jnp.maximum(0, state.enemy.health_current - unit.ranged_base_attack_damage)
            unit_action_points = unit.action_points_current - 1
            new_target = target.replace(health_current=jnp.float32(target_health_after))
            new_unit = unit.replace(action_points_current=jnp.float32(unit_action_points))
            
            def do_ranged_attack_player():
                return state.replace(
                player=new_unit,
                enemy=new_target,
                steps=jnp.int32(state.steps + 1),
            )
            def do_ranged_attack_enemy():
                return state.replace(
                player=new_target,
                enemy=new_unit,
            )

            new_state = lax.cond(
                jnp.equal(state.player.unit_id, unit.unit_id),
                do_ranged_attack_player,
                do_ranged_attack_enemy,
            )

            return new_state

        def invalid_move(_):
            # debug.print("invalid move: ranged attack, unit: {unit}", unit = unit.unit_id)
            return do_invalid_move(state, unit, target)

        return lax.cond( # TODO USE LAX SWITCH ONCE MORE UNITS
            self.is_valid(state, unit, target),
            do_ranged_attack,
            invalid_move,
            None,
        )

class EndTurnAction:
    def __init__(self):
        self.num_agents = 2

    def is_valid(self, state, unit, target):
        # End turn action is always valid
        return True

    def execute(self, state, unit, target):
        # if player (split this out into a player_end_turn and enemy_end_turn function that you then call in this if)
        def end_turn_player():
            new_player_state = state.player.replace(
                action_points_current=jnp.float32(state.player.action_points_max),
                movement_points_current=jnp.float32(state.player.movement_points_max),
                )
            aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            aidx = (aidx + 1) % self.num_agents
            cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)
            return state.replace(
                player=new_player_state,
                steps=jnp.int32(state.steps + 1),
                turn_count=jnp.int32(state.turn_count + 1),
                cur_player_idx=cur_player_idx,
                )
        def end_turn_enemy():
            new_enemy_state = state.enemy.replace(
                action_points_current=jnp.float32(state.enemy.action_points_max),
                movement_points_current=jnp.float32(state.enemy.movement_points_max),
                )
            aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            aidx = (aidx + 1) % self.num_agents
            cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)
            return state.replace(
                enemy=new_enemy_state,
                steps=jnp.int32(state.steps + 1),
                cur_player_idx=cur_player_idx,
                )

        return lax.cond(
            jnp.equal(state.player.unit_id, unit.unit_id),
            end_turn_player,
            end_turn_enemy,
            )