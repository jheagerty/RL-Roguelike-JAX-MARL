import jax
import jax.numpy as jnp
from jax import lax
from config import env_config
from utils import euclidean_distance

@jax.jit
def reward_function(old_state, new_state, action):
    """Calculate rewards for both players based on state changes.
    
    Args:
        old_state: Previous game state
        new_state: Current game state 
        action: Action taken
        
    Returns:
        Tuple[float, float]: (player_reward, enemy_reward) containing rewards for each team
    """
    # Initialize base rewards from damage dealt
    player_reward = jnp.float32(0)
    enemy_reward = jnp.float32(0)

    # Calculate health changes
    old_team1_health = jnp.sum(old_state.units.health[:env_config['HEROES_PER_TEAM'], 0])
    old_team2_health = jnp.sum(old_state.units.health[env_config['HEROES_PER_TEAM']:, 0])
    new_team1_health = jnp.sum(new_state.units.health[:env_config['HEROES_PER_TEAM'], 0])
    new_team2_health = jnp.sum(new_state.units.health[env_config['HEROES_PER_TEAM']:, 0])
    
    # Base rewards from damage
    player_reward += old_team2_health - new_team2_health
    enemy_reward += old_team1_health - new_team1_health

    # Calculate distances between teams
    def get_team_distances(state):
        """Calculate min distance between teams."""
        team1_pos = state.units.location[:env_config['HEROES_PER_TEAM']]
        team2_pos = state.units.location[env_config['HEROES_PER_TEAM']:]
        
        def calc_unit_distances(pos1):
            def calc_distance(pos2):
                return euclidean_distance(pos1[0], pos1[1], pos2[0], pos2[1])
            return jax.vmap(calc_distance)(team2_pos)
            
        all_distances = jax.vmap(calc_unit_distances)(team1_pos)
        return jnp.min(all_distances)

    old_distance = get_team_distances(old_state)
    new_distance = get_team_distances(new_state)
    
    # Movement reward functions
    def calc_movement_reward(old_dist, new_dist, initial_dist):
        """Calculate reward for movement relative to initial distance."""
        improvement = old_dist - new_dist
        return 10.0 * improvement / (initial_dist + 1.0)
    
    # Apply movement rewards
    def apply_movement_reward(reward, old_dist, new_dist, initial_dist):
        return lax.cond(
            new_dist < old_dist,
            lambda _: reward + calc_movement_reward(old_dist, new_dist, initial_dist),
            lambda _: reward,
            operand=None
        )
    
    # Calculate if units are in attack range of enemies
    def check_attack_ranges(state, unit_idx):
        """Check if unit is within any enemy attack range."""
        unit_pos = state.units.location[unit_idx]
        enemy_start = env_config['HEROES_PER_TEAM'] if unit_idx < env_config['HEROES_PER_TEAM'] else 0
        enemy_end = enemy_start + env_config['HEROES_PER_TEAM']
        enemy_positions = state.units.location[enemy_start:enemy_end]
        
        def check_ranges(enemy_pos):
            distance = euclidean_distance(
                unit_pos[0], unit_pos[1],
                enemy_pos[0], enemy_pos[1]
            )
            # Check both melee and ranged attack ranges
            in_melee = distance <= state.units.melee_attack[enemy_start:enemy_end, 1]  # attack_range index
            in_ranged = distance <= state.units.ranged_attack[enemy_start:enemy_end, 1]  # attack_range index
            return jnp.logical_or(in_melee, in_ranged)
            
        return jnp.any(jax.vmap(check_ranges)(enemy_positions))

    def check_unit_moved(old_state, new_state, unit_idx):
        """Check if unit changed position."""
        old_pos = old_state.units.location[unit_idx]
        new_pos = new_state.units.location[unit_idx]
        return jnp.logical_or(
            old_pos[0] != new_pos[0],
            old_pos[1] != new_pos[1]
        )

    def apply_no_move_penalty(reward, old_state, new_state, unit_idx, action):
        """Apply penalty if unit didn't move when it could have."""
        # Check if action was a movement action (typically indices 0-7)
        is_move_action = action < 8  # Assuming first 8 actions are movement
        
        # Check if unit moved
        unit_moved = check_unit_moved(old_state, new_state, unit_idx)
        
        # Check if unit is in range of any enemy
        in_attack_range = check_attack_ranges(old_state, unit_idx)
        
        # Apply penalty if:
        # 1. Movement action was chosen
        # 2. Unit didn't actually move
        # 3. Unit wasn't in attack range
        should_penalize = jnp.logical_and(
            is_move_action,
            jnp.logical_and(
                ~unit_moved,
                ~in_attack_range
            )
        )
        
        return lax.cond(
            should_penalize,
            lambda r: r - 1.0,  # Apply penalty
            lambda r: r,        # No penalty
            reward
        )
    
    # Update rewards based on movement
    player_reward = apply_movement_reward(
        player_reward, 
        old_distance,
        new_distance, 
        new_state.initial_distance
    )
    
    enemy_reward = apply_movement_reward(
        enemy_reward,
        old_distance,
        new_distance,
        new_state.initial_distance
    )

    # Victory/defeat reward functions
    def bonus_for_enemy_defeated(current_reward):
        return current_reward + (70 / (new_state.turn_count + 1))
    
    def penalty_for_player_defeated(current_reward):
        return current_reward - (50 / (new_state.turn_count + 1))

    # Apply victory/defeat rewards
    player_reward = lax.cond(
        new_team2_health <= 0,
        lambda r: bonus_for_enemy_defeated(r),
        lambda r: r,
        player_reward
    )
    
    enemy_reward = lax.cond(
        new_team1_health <= 0,
        lambda r: bonus_for_enemy_defeated(r),
        lambda r: r,
        enemy_reward
    )

    # Apply defeat penalties
    player_reward = lax.cond(
        new_team1_health <= 0,
        lambda r: penalty_for_player_defeated(r),
        lambda r: r,
        player_reward
    )
    
    enemy_reward = lax.cond(
        new_team2_health <= 0,
        lambda r: penalty_for_player_defeated(r),
        lambda r: r,
        enemy_reward
    )

    # Apply no-move penalties to both teams
    player_unit_idx = 0  # First unit of team 1
    enemy_unit_idx = env_config['HEROES_PER_TEAM']  # First unit of team 2
    
    player_reward = apply_no_move_penalty(
        player_reward,
        old_state,
        new_state,
        player_unit_idx,
        action
    )
    
    enemy_reward = apply_no_move_penalty(
        enemy_reward,
        old_state,
        new_state,
        enemy_unit_idx,
        action
    )

    return player_reward, enemy_reward

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

    # return player_reward, enemy_reward

