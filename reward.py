import jax
import jax.numpy as jnp
from jax import lax

def reward_function(old_state, new_state, action):
    player_reward = jnp.float32(0)
    enemy_reward = jnp.float32(0)

    # # Calculate the basic reward based on enemy health change.
    player_reward = old_state.enemy.health_current - new_state.enemy.health_current
    enemy_reward = old_state.player.health_current - new_state.player.health_current

    # No reward.
    def no_reward(current_reward):
        return current_reward

    # Reward for defeating an enemy.
    def bonus_for_enemy_defeated(current_reward):
        # jax.debug.print("enemy dead")
        return current_reward + (70 / (new_state.turn_count + 1))
    
    # Apply reward for defeating an enemy.
    player_reward = lax.cond(
        new_state.enemy.health_current <= 0,
        lambda _: bonus_for_enemy_defeated(player_reward),
        lambda _: no_reward(player_reward),
        None,
    )
    enemy_reward = lax.cond(
        new_state.player.health_current <= 0,
        lambda _: bonus_for_enemy_defeated(enemy_reward),
        lambda _: no_reward(enemy_reward),
        None,
    )

    # Negative reward on player defeat
    def penalty_for_player_defeated(current_reward):
        # jax.debug.print("player dead")
        return current_reward - (50 / (new_state.turn_count + 1))

   # Apply negative reward for being defeated.
    player_reward = lax.cond(
        jax.numpy.equal(new_state.player.health_current, 0),
        lambda _: penalty_for_player_defeated(player_reward),
        lambda _: no_reward(player_reward),
        None,
    )
    enemy_reward = lax.cond(
        jax.numpy.equal(new_state.enemy.health_current, 0),
        lambda _: penalty_for_player_defeated(enemy_reward),
        lambda _: no_reward(enemy_reward),
        None,
    )

    # # Negative reward for not moving when you can and are out of range of the enemy.
    # def penalty_for_not_moving(current_reward):
    #     return reward - 1

    # # Apply negative reward for not moving.
    # is_in_action_range = jnp.logical_and(0 <= action, action <= 7)
    # player_did_not_move = jnp.logical_and(
    #     old_state.player.location_x == new_state.player.location_x,
    #     old_state.player.location_y == new_state.player.location_y
    # )
    # should_penalty_apply = jnp.logical_and(is_in_action_range, player_did_not_move)
    # reward = lax.cond(should_penalty_apply,
    #                 lambda _: penalty_for_not_moving(reward),
    #                 lambda _: no_reward(reward),
    #                 None)

    # # Reward for moving closer to the enemy.
    # def bonus_for_closer(current_reward):
    #     distance_improvement = old_state.previous_closest_distance - new_state.distance_to_enemy
    #     return reward + 10 * distance_improvement / (new_state.initial_distance - 1)

    # # Apply reward bonus for moving closer to the enemy.
    # moved_closer_to_enemy = new_state.distance_to_enemy < old_state.previous_closest_distance
    # reward = lax.cond(moved_closer_to_enemy,
    #                 lambda _: bonus_for_closer(reward),
    #                 lambda _: no_reward(reward),
    #                 None)

    return player_reward, enemy_reward

