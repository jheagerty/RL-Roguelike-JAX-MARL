# base_actions.py
import jax.numpy as jnp
import jax
from jax import lax
import chex
from typing import List, Callable

from utils import euclidean_distance, is_within_bounds, is_collision, do_attack
from actions import Action
from data_classes import GameState, AttackType, DamageType
from config import env_config

# Create global registry
base_action_registry = {}

def register_action(name: str, create_fn: Callable) -> None:
    """Register an action creation function.
    
    Args:
        name: Identifier for the action
        create_fn: Function that creates action instances
    """
    base_action_registry[name] = create_fn

# Register movement actions for each direction
register_action(
    "MoveAction",
    lambda: [MoveAction(dx, dy) 
             for dx in (-1, 0, 1) 
             for dy in (-1, 0, 1) 
             if dx != 0 or dy != 0]
)

# Register other base actions
register_action("MeleeAttackAction", lambda: [MeleeAttackAction()])
register_action("RangedAttackAction", lambda: [RangedAttackAction()])
register_action("EndTurnAction", lambda: [EndTurnAction()])
register_action(
    "PickPoolAbility",
    lambda: [PickPoolAbility(i) for i in range(env_config['ABILITY_POOL_SIZE'])]
)

class PickPoolAbility(Action):
    """Action that lets a unit pick an ability from the pool."""
    def __init__(self, pool_index: int):
        super().__init__()
        self.pool_index = pool_index
        self.num_agents = 2
        self._ability_description = f"Pick ability {pool_index} from pool"

    def _is_valid_ability(
        self,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int = jnp.int32(-1),
    ) -> chex.Array:
        """Check if ability can be picked from pool.
        
        Args:
            state: Current game state
            source_id: ID of unit attempting to pick
            target_id: Unused for picking
            ability_slot: Unused for picking
            
        Returns:
            Boolean indicating if ability can be picked
        """
        return jnp.logical_and(
            state.pick_mode == 1,
            state.ability_pool_picked[self.pool_index] == 0
        )

    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int = jnp.int32(-1),
    ) -> GameState:
        """Execute ability pick logic.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            source_id: ID of unit picking ability
            target_id: Unused for picking
            ability_slot: Unused for picking
            
        Returns:
            Updated game state with new ability assigned
        """
        # Get ability parameters from pool
        pool_ability = state.ability_pool[self.pool_index, :]

        # Update unit's second ability slot (slot 1)
        new_abilities = state.units.abilities.at[source_id, 1].set(
            jnp.array([
                pool_ability[0],  # ability_index
                pool_ability[1],  # base_cooldown
                pool_ability[2],  # current_cooldown
                pool_ability[3],  # parameter_1
                pool_ability[4],  # parameter_2
                pool_ability[5]   # parameter_3
            ])
        )

        # Set action points to 0 to end turn
        new_action_points = state.units.action_points.at[source_id, 1].set(0)

        # Calculate new pick state
        new_pick_count = state.pick_count + 1
        new_pick_mode = jnp.where(new_pick_count >= 2, 0, 1)
        
        # Update current player
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        aidx = (aidx + 1) % self.num_agents
        new_cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)

        # Create updated units state
        new_units = state.units.replace(
            abilities=new_abilities,
            action_points=new_action_points
        )

        # Mark ability as picked in pool
        new_ability_pool_picked = state.ability_pool_picked.at[self.pool_index].set(1)

        # Return updated game state
        return state.replace(
            units=new_units,
            pick_count=new_pick_count,
            pick_mode=new_pick_mode,
            cur_player_idx=new_cur_player_idx,
            ability_pool_picked=new_ability_pool_picked
        )
    
class MoveAction(Action):
    """Action that moves a unit in a specified direction."""
    def __init__(self, dx: int, dy: int):
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

    def _is_valid_ability(self, state, source_id, target_id, ability_slot):
        # Check pick mode first
        not_pick_mode = state.pick_mode == 0
        
        # Early return if in pick mode
        return lax.cond(
            not_pick_mode,
            lambda _: self._check_movement_validity(state, source_id, target_id),
            lambda _: jnp.array(False),
            operand=None
        )
    
    def _check_movement_validity(self, state, source_id, target_id):
        # Original movement validation logic
        new_x = state.units.location[source_id, 0] + self.dx
        new_y = state.units.location[source_id, 1] + self.dy
        
        within_bounds = is_within_bounds(new_x, new_y)
        no_collision = ~jnp.any(jnp.all(
            state.units.location == jnp.array([new_x, new_y]), 
            axis=1
        ))
        
        movement_cost = jnp.float32(jnp.sqrt(self.dx**2 + self.dy**2))
        enough_movement_points = state.units.movement_points[source_id, 1] >= movement_cost

        return jnp.logical_and(within_bounds, jnp.logical_and(no_collision, enough_movement_points))

    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int = jnp.int32(-1)
    ) -> GameState:
        """Execute movement logic.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            source_id: ID of unit moving
            target_id: ID of target location (unused)
            ability_slot: Unused for movement
            
        Returns:
            Updated game state with new unit position
        """
        # Calculate new position and movement cost
        new_x = state.units.location[source_id, 0] + self.dx
        new_y = state.units.location[source_id, 1] + self.dy
        distance = jnp.float32(jnp.sqrt(self.dx**2 + self.dy**2))
        
        # Update unit location
        new_location = state.units.location.at[source_id].set(
            jnp.array([new_x, new_y])
        )
        
        # Update movement points
        new_movement_points = state.units.movement_points.at[source_id, 1].set(
            state.units.movement_points[source_id, 1] - distance
        )
        
        # Calculate new distance between teams TODO: update to become distances matrix
        # Take minimum distance between any pair of units from opposing teams
        team_size = env_config['HEROES_PER_TEAM']
        team1_locs = new_location[:team_size]
        team2_locs = new_location[team_size:]
        
        distances = jax.vmap(
            lambda t1: jax.vmap(
                lambda t2: euclidean_distance(t1[0], t1[1], t2[0], t2[1])
            )(team2_locs)
        )(team1_locs)
        
        new_distance = jnp.min(distances)
        
        # Create updated units state
        new_units = state.units.replace(
            location=new_location,
            movement_points=new_movement_points
        )
        
        # Update game state
        return state.replace(
            units=new_units,
            distance_to_enemy=new_distance,
            previous_closest_distance=jnp.minimum(
                state.previous_closest_distance,
                new_distance
            )
        )

class MeleeAttackAction(Action):
    def __init__(self):
        super().__init__()
        self._ability_description = "Basic melee attack"

    def _is_valid_ability(self, state, source_id, target_id, ability_slot):
        not_pick_mode = state.pick_mode == 0
        
        return lax.cond(
            not_pick_mode,
            lambda _: self._check_attack_validity(state, source_id, target_id),
            lambda _: jnp.array(False),
            operand=None
        )
    
    def _check_attack_validity(self, state, source_id, target_id):
        enough_ap = state.units.action_points[source_id, 1] >= 1
        dist = euclidean_distance(
            state.units.location[source_id, 0],
            state.units.location[source_id, 1],
            state.units.location[target_id, 0],
            state.units.location[target_id, 1]
        )
        within_range = dist <= state.units.melee_attack[source_id, 1]
        return jnp.logical_and(enough_ap, within_range)

    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int = jnp.int32(-1),
    ) -> GameState:
        """Execute melee attack logic.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            source_id: ID of attacking unit
            target_id: ID of target unit
            ability_slot: Unused for base attacks
            
        Returns:
            Updated game state after attack
        """
        # Reduce action points
        new_action_points = state.units.action_points.at[source_id, 1].set(
            state.units.action_points[source_id, 1] - 1
        )

        # Increment attack counter
        new_melee_count = state.units.base_melee_attack_count.at[source_id].add(1)

        # Perform attack calculations using do_attack utility
        state = do_attack(
            state,
            source_id,
            target_id,
            AttackType.MELEE,
            DamageType.PHYSICAL
        )

        # Update state with new values
        new_units = state.units.replace(
            action_points=new_action_points,
            base_melee_attack_count=new_melee_count
        )

        return state.replace(units=new_units)

class RangedAttackAction(Action):
    def __init__(self):
        super().__init__()
        self._ability_description = "Basic ranged attack"

    def _is_valid_ability(self, state, source_id, target_id, ability_slot):
        not_pick_mode = state.pick_mode == 0
        
        return lax.cond(
            not_pick_mode,
            lambda _: self._check_attack_validity(state, source_id, target_id),
            lambda _: jnp.array(False),
            operand=None
        )
    
    def _check_attack_validity(self, state, source_id, target_id):
        enough_ap = state.units.action_points[source_id, 1] >= 1
        dist = euclidean_distance(
            state.units.location[source_id, 0],
            state.units.location[source_id, 1],
            state.units.location[target_id, 0],
            state.units.location[target_id, 1]
        )
        within_range = dist <= state.units.ranged_attack[source_id, 1]
        return jnp.logical_and(enough_ap, within_range)

    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int = jnp.int32(-1),
    ) -> GameState:
        """Execute ranged attack logic.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            source_id: ID of attacking unit
            target_id: ID of target unit
            ability_slot: Unused for base attacks
            
        Returns:
            Updated game state after attack
        """
        # Reduce action points
        new_action_points = state.units.action_points.at[source_id, 1].set(
            state.units.action_points[source_id, 1] - 1
        )

        # Increment attack counter
        new_ranged_count = state.units.base_ranged_attack_count.at[source_id].add(1)

        # Perform attack calculations using do_attack utility
        state = do_attack(
            state,
            source_id,
            target_id,
            AttackType.RANGED,
            DamageType.PHYSICAL
        )

        # Update state with new values
        new_units = state.units.replace(
            action_points=new_action_points,
            base_ranged_attack_count=new_ranged_count
        )

        return state.replace(units=new_units)

class EndTurnAction(Action):

    def _is_valid_ability(self, state, source_id, target_id, ability_slot):
        # Only valid when not in pick mode
        not_pick_mode = state.pick_mode == 0
        
        return lax.cond(
            not_pick_mode,
            lambda _: jnp.array(True),  # Always valid if not in pick mode
            lambda _: jnp.array(False),
            operand=None
        )
    
    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int = jnp.int32(-1),
    ) -> GameState:
        """Execute end turn logic.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            source_id: ID of unit ending turn
            target_id: Unused for end turn
            ability_slot: Unused for base actions
            
        Returns:
            Updated game state with refreshed resources and reduced cooldowns
        """
        # Existing end turn logic...
        new_action_points = state.units.action_points.at[source_id, 1].set(
            state.units.action_points[source_id, 2]  # Set current to max
        )
        
        new_movement_points = state.units.movement_points.at[source_id, 1].set(
            state.units.movement_points[source_id, 2]  # Set current to max
        )

        # Reduce cooldowns for all abilities by 1 (not below 0)
        abilities = state.units.abilities
        new_cooldowns = jnp.maximum(
            0,
            abilities[source_id, :, 2] - 1  # Reduce current_cooldown by 1
        )
        new_abilities = abilities.at[source_id, :, 2].set(new_cooldowns)

        # Remaining end turn logic...
        new_end_turn_count = state.units.end_turn_count.at[source_id].add(1)

        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        aidx = (aidx + 1) % (env_config['HEROES_PER_TEAM'] * 2)
        new_cur_player_idx = jnp.zeros(env_config['HEROES_PER_TEAM'] * 2).at[aidx].set(1)

        new_units = state.units.replace(
            action_points=new_action_points,
            movement_points=new_movement_points,
            abilities=new_abilities,
            end_turn_count=new_end_turn_count
        )

        def increment_turn(_):
            return state.turn_count + 1

        def keep_turn(_):
            return state.turn_count

        new_turn_count = lax.cond(
            source_id < env_config['HEROES_PER_TEAM'],
            increment_turn,
            keep_turn,
            None
        )

        return state.replace(
            units=new_units,
            turn_count=new_turn_count,
            cur_player_idx=new_cur_player_idx
        )