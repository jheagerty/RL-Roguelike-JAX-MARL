# ability_actions.py
import jax.numpy as jnp
import chex
from jax import lax
from utils import euclidean_distance, do_damage, do_attack
from actions import Action
from data_classes import GameState, AttackType, DamageType
from config import env_config

# Create global registry
ability_registry = {}

def register_ability(name, create_fn):
    """Register an ability creation function."""
    ability_registry[name] = create_fn

class SuicideAction(Action):
    def __init__(self):
        super().__init__()
        self._ability_description = "Deal damage based on strength to target and self"
        self._base_cooldown = jnp.int32(3)
        self._parameter_1 = jnp.float32(8)  # range
        self._parameter_2 = jnp.float32(0)  # base_damage
        self._parameter_3 = jnp.float32(3)  # strength multiplier

    def _is_valid_ability(
        self,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int,
    ) -> chex.Array:
        """Check if suicide ability can be used.
        
        Args:
            state: Current game state
            source_id: ID of unit attempting to use ability
            target_id: ID of target unit
            ability_slot: Index of ability being used
            
        Returns:
            Boolean indicating if ability can be used
        """
        # Get ability parameters from state
        ability_params = state.units.abilities[source_id, ability_slot]
        ability_range = ability_params[3]  # parameter_1 stores range

        # Calculate distance between units 
        dist = euclidean_distance(
            state.units.location[source_id, 0],
            state.units.location[source_id, 1],
            state.units.location[target_id, 0],
            state.units.location[target_id, 1]
        )

        enough_ap = state.units.action_points[source_id, 1] >= 1
        within_range = dist <= ability_range
        not_pick_mode = state.pick_mode == 0

        return jnp.logical_and(
            jnp.logical_and(enough_ap, within_range),
            not_pick_mode
        )

    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int,
    ) -> GameState:
        """Execute suicide ability logic.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            source_id: ID of unit using ability
            target_id: ID of target unit
            ability_slot: Index of ability being used
            
        Returns:
            Updated game state after ability effects
        """
        # Get ability parameters
        ability_params = state.units.abilities[source_id, ability_slot]
        base_damage = ability_params[4]  # parameter_2
        strength_mult = ability_params[5]  # parameter_3

        # Calculate damage based on source unit's strength
        damage = base_damage + (strength_mult * state.units.strength[source_id, 1])

        # Apply damage to target using magical damage
        new_state = do_damage(
            state,
            source_id,
            target_id,
            damage,
            DamageType.MAGICAL,
            return_damage=False
        )

        # Apply self-damage as magical damage
        new_state = do_damage(
            new_state,
            source_id,
            source_id,
            base_damage,  # Self damage uses base value only
            DamageType.MAGICAL,
            return_damage=False
        )

        # Update action points and cooldown
        new_action_points = new_state.units.action_points.at[source_id, 1].set(
            new_state.units.action_points[source_id, 1] - 1
        )

        new_abilities = new_state.units.abilities.at[source_id, ability_slot, 2].set(
            new_state.units.abilities[source_id, ability_slot, 1]  # Set current cd to base cd
        )

        # Increment usage counter
        new_suicide_count = new_state.units.suicide_ability_count.at[source_id].add(1)

        # Create updated units state
        new_units = new_state.units.replace(
            action_points=new_action_points,
            abilities=new_abilities,
            suicide_ability_count=new_suicide_count
        )

        # Return updated game state
        return new_state.replace(units=new_units)

# Register ability
register_ability("SuicideAction", lambda: [SuicideAction()])

# Steal Strength - reduce enemy strength and increase own strength
# Register the new action
register_ability("StealStrengthAction", lambda: [StealStrengthAction()])

class StealStrengthAction(Action):
    def __init__(self):
        super().__init__()
        self._ability_description = "Steal strength from target unit"
        self._base_cooldown = jnp.int32(1)
        self._parameter_1 = jnp.float32(4)  # range
        self._parameter_2 = jnp.float32(2)  # strength_steal_amount

    def _is_valid_ability(
        self, 
        state: GameState, 
        source_id: int, 
        target_id: int,
        ability_slot: int,
    ) -> chex.Array:
        """Check if strength steal ability can be used.
        
        Args:
            state: Current game state
            source_id: ID of unit attempting to use ability
            target_id: ID of targeted unit
            
        Returns:
            Boolean array indicating if ability can be used
        """
        # Extract ability parameters from source unit's ability data
        ability_params = state.units.abilities[source_id, ability_slot]
        ability_range = ability_params[3]  # parameter_1 stores range
        
        # Check action point cost
        enough_ap = state.units.action_points[source_id, 1] >= 1
        
        # Calculate distance between source and target
        dist = euclidean_distance(
            state.units.location[source_id, 0],
            state.units.location[source_id, 1],
            state.units.location[target_id, 0], 
            state.units.location[target_id, 1]
        )
        
        # Validate range and game state
        within_range = dist <= ability_range
        not_pick_mode = state.pick_mode == 0
        
        # Combine all conditions
        return jnp.logical_and(
            jnp.logical_and(enough_ap, within_range),
            not_pick_mode
        )

    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int,
    ) -> GameState:
        """Execute strength steal ability logic.
        
        Args:
            key: PRNG key for any randomization
            state: Current game state
            source_id: ID of unit using the ability
            target_id: ID of unit being targeted
            ability_slot: Index of ability being used
            
        Returns:
            Updated game state with modified strength values and ability cooldown
        """
        # Get ability parameters from source unit
        ability_params = state.units.abilities[source_id, ability_slot]
        steal_amount = ability_params[4]  # parameter_2 stores steal amount
        
        # Update strength values for both units
        new_strength = state.units.strength.at[target_id, 1].set(
            jnp.maximum(0, state.units.strength[target_id, 1] - steal_amount)
        ).at[source_id, 1].set(
            state.units.strength[source_id, 1] + steal_amount
        )
        
        # Deduct action point cost
        new_action_points = state.units.action_points.at[source_id, 1].set(
            state.units.action_points[source_id, 1] - 1
        )
        
        # Set ability on cooldown
        new_abilities = state.units.abilities.at[source_id, ability_slot, 2].set(
            state.units.abilities[source_id, ability_slot, 1]  # current_cd = base_cd
        )

        # Create updated unit state
        new_units = state.units.replace(
            strength=new_strength,
            action_points=new_action_points,
            abilities=new_abilities
        )

        # Return updated game state
        return state.replace(units=new_units)

class MultiAttackAction(Action):
    def __init__(self):
        super().__init__()
        self._ability_description = "Perform three ranged attacks in one action"
        self._base_cooldown = jnp.int32(1)
        self._parameter_1 = jnp.float32(8)  # range

    def _is_valid_ability(
        self,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int
    ) -> chex.Array:
        """Check if multi-attack ability can be used."""
        # Get ability parameters
        ability_params = state.units.abilities[source_id, ability_slot]
        ability_range = ability_params[3]  # parameter_1 stores range

        # Calculate distance between units
        dist = euclidean_distance(
            state.units.location[source_id, 0],
            state.units.location[source_id, 1],
            state.units.location[target_id, 0],
            state.units.location[target_id, 1]
        )

        enough_ap = state.units.action_points[source_id, 1] >= 1
        within_range = dist <= ability_range
        not_pick_mode = state.pick_mode == 0

        return jnp.logical_and(
            jnp.logical_and(enough_ap, within_range),
            not_pick_mode
        )

    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int
    ) -> GameState:
        """Execute three ranged attacks in sequence."""
        # Do three ranged attacks
        new_state = do_attack(state, source_id, target_id, AttackType.RANGED, DamageType.PHYSICAL)
        new_state = do_attack(new_state, source_id, target_id, AttackType.RANGED, DamageType.PHYSICAL)
        new_state = do_attack(new_state, source_id, target_id, AttackType.RANGED, DamageType.PHYSICAL)

        # Update action points and cooldown
        new_action_points = new_state.units.action_points.at[source_id, 1].set(
            new_state.units.action_points[source_id, 1] - 1
        )

        new_abilities = new_state.units.abilities.at[source_id, ability_slot, 2].set(
            new_state.units.abilities[source_id, ability_slot, 1]  # Set current cd to base cd
        )

        # Increment usage counter
        new_multi_attack_count = new_state.units.multi_attack_ability_count.at[source_id].add(1)

        # Create updated units state
        new_units = new_state.units.replace(
            action_points=new_action_points,
            abilities=new_abilities,
            multi_attack_ability_count=new_multi_attack_count
        )

        # Return updated game state
        return new_state.replace(units=new_units)

# Register ability
register_ability("MultiAttackAction", lambda: [MultiAttackAction()])

class ReturnAction(Action):
    def __init__(self):
        super().__init__()
        self._ability_description = "Set physical damage return to current strength"
        self._base_cooldown = jnp.int32(10)
        self._parameter_1 = jnp.float32(0)  # No range needed - self cast

    def _is_valid_ability(
        self,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int,
    ) -> chex.Array:
        """Check if return ability can be used.
        
        Args:
            state: Current game state
            source_id: ID of unit attempting to use ability
            target_id: ID of targeted unit (ignored - self cast)
            ability_slot: Index of ability being used
            
        Returns:
            Boolean array indicating if ability can be used
        """
        enough_ap = state.units.action_points[source_id, 1] >= 1
        not_pick_mode = state.pick_mode == 0
        return jnp.logical_and(enough_ap, not_pick_mode)

    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int,
    ) -> GameState:
        """Execute return ability logic.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            source_id: ID of unit using ability
            target_id: ID of target unit (ignored - self cast)
            ability_slot: Index of ability being used
            
        Returns:
            Updated game state with modified damage return and cooldown
        """
        # Set physical damage return to current strength
        new_physical_defence = state.units.physical_defence.at[source_id, 4].set(
            state.units.strength[source_id, 1]  # damage_return = current_strength
        )

        # Update action points and cooldown
        new_action_points = state.units.action_points.at[source_id, 1].set(
            state.units.action_points[source_id, 1] - 1
        )

        new_abilities = state.units.abilities.at[source_id, ability_slot, 2].set(
            state.units.abilities[source_id, ability_slot, 1]  # Set current cd to base cd
        )

        # Increment usage counter
        new_return_count = state.units.return_ability_count.at[source_id].add(1)

        # Create updated units state
        new_units = state.units.replace(
            physical_defence=new_physical_defence,
            action_points=new_action_points,
            abilities=new_abilities,
            return_ability_count=new_return_count
        )

        # Return updated game state
        return state.replace(units=new_units)

# Register ability
register_ability("ReturnAction", lambda: [ReturnAction()])

class StrengthRegenAction(Action):
    def __init__(self):
        super().__init__()
        self._ability_description = "Heal based on current strength"
        self._base_cooldown = jnp.int32(10)
        self._parameter_1 = jnp.float32(5)  # healing multiplier

    def _is_valid_ability(
        self,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int,
    ) -> chex.Array:
        """Check if strength regen ability can be used.
        
        Args:
            state: Current game state
            source_id: ID of unit attempting to use ability
            target_id: ID of targeted unit (ignored - self cast)
            ability_slot: Index of ability being used
            
        Returns:
            Boolean array indicating if ability can be used
        """
        enough_ap = state.units.action_points[source_id, 1] >= 1
        not_pick_mode = state.pick_mode == 0
        return jnp.logical_and(enough_ap, not_pick_mode)

    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int,
    ) -> GameState:
        """Execute strength-based healing logic.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            source_id: ID of unit using ability
            target_id: ID of target unit (ignored - self cast)
            ability_slot: Index of ability being used
            
        Returns:
            Updated game state with modified health and cooldown
        """
        # Get ability parameters
        ability_params = state.units.abilities[source_id, ability_slot]
        healing_mult = ability_params[3]  # parameter_1 stores healing multiplier
        
        # Calculate healing based on current strength
        healing = state.units.strength[source_id, 1] * healing_mult
        
        # Update health (capped at max health)
        new_health = state.units.health.at[source_id, 0].set(
            jnp.minimum(
                state.units.health[source_id, 1],  # max health
                state.units.health[source_id, 0] + healing  # current + healing
            )
        )

        # Update action points and cooldown
        new_action_points = state.units.action_points.at[source_id, 1].set(
            state.units.action_points[source_id, 1] - 1
        )

        new_abilities = state.units.abilities.at[source_id, ability_slot, 2].set(
            state.units.abilities[source_id, ability_slot, 1]  # Set current cd to base cd
        )

        # Increment usage counter
        new_strength_regen_count = state.units.strength_regen_ability_count.at[source_id].add(1)

        # Create updated units state
        new_units = state.units.replace(
            health=new_health,
            action_points=new_action_points,
            abilities=new_abilities,
            strength_regen_ability_count=new_strength_regen_count
        )

        # Return updated game state
        return state.replace(units=new_units)

# Register ability
register_ability("StrengthRegenAction", lambda: [StrengthRegenAction()])

class AddBarrierAction(Action):
    def __init__(self):
        super().__init__()
        self._ability_description = "Add barrier based on current resolve"
        self._base_cooldown = jnp.int32(20)
        self._parameter_1 = jnp.float32(5)  # barrier multiplier

    def _is_valid_ability(
        self,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int,
    ) -> chex.Array:
        """Check if barrier ability can be used.
        
        Args:
            state: Current game state
            source_id: ID of unit attempting to use ability
            target_id: ID of targeted unit (ignored - self cast)
            ability_slot: Index of ability being used
            
        Returns:
            Boolean array indicating if ability can be used
        """
        enough_ap = state.units.action_points[source_id, 1] >= 1
        not_pick_mode = state.pick_mode == 0
        return jnp.logical_and(enough_ap, not_pick_mode)

    def _perform_action(
        self,
        key: chex.PRNGKey,
        state: GameState,
        source_id: int,
        target_id: int,
        ability_slot: int
    ) -> GameState:
        """Execute barrier adding logic.
        
        Args:
            key: PRNG key for randomization
            state: Current game state
            source_id: ID of unit using ability
            target_id: ID of target unit (ignored - self cast)
            ability_slot: Index of ability being used
            
        Returns:
            Updated game state with modified barrier and cooldown
        """
        # Get ability parameters
        ability_params = state.units.abilities[source_id, ability_slot]
        barrier_mult = ability_params[3]  # parameter_1 stores barrier multiplier

        # Calculate barrier amount based on resolve
        barrier_amount = state.units.resolve[source_id, 1] * barrier_mult

        # Update barrier (capped at max barrier)
        new_barrier = state.units.barrier.at[source_id, 0].set(
            jnp.minimum(
                state.units.barrier[source_id, 1],  # max barrier
                state.units.barrier[source_id, 0] + barrier_amount  # current + new
            )
        )

        # Update action points and cooldown
        new_action_points = state.units.action_points.at[source_id, 1].set(
            state.units.action_points[source_id, 1] - 1
        )

        new_abilities = state.units.abilities.at[source_id, ability_slot, 2].set(
            state.units.abilities[source_id, ability_slot, 1]  # Set current cd to base cd
        )

        # Increment usage counter
        new_add_barrier_count = state.units.add_barrier_ability_count.at[source_id].add(1)

        # Create updated units state
        new_units = state.units.replace(
            barrier=new_barrier,
            action_points=new_action_points,
            abilities=new_abilities,
            add_barrier_ability_count=new_add_barrier_count
        )

        # Return updated game state
        return state.replace(units=new_units)

# Register ability
register_ability("AddBarrierAction", lambda: [AddBarrierAction()])


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