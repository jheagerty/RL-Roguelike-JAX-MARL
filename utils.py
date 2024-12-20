import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import replace
import os
from pathlib import Path
import chex
from data_classes import AttackType, DamageType, GameState

# Util Functions
def euclidean_distance(x1, y1, x2, y2):
    return jnp.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def is_within_bounds(x, y, max_bound=20):
    return jnp.logical_and(jnp.logical_and(0 <= x, x < max_bound),
                           jnp.logical_and(0 <= y, y < max_bound))

def is_collision(x1, y1, x2, y2):
    return jnp.logical_and(x1 == x2, y1 == y2)

def generate_unique_pairs(key, batch_size=10):
    # Generate the first pair
    key, subkey = jax.random.split(key)
    x1, y1 = jax.random.randint(subkey, (2,), 0, 20, dtype=jnp.int32)

    # Function to generate a batch of random pairs
    def generate_batch(key):
        key, subkey = jax.random.split(key)
        pairs = jax.random.randint(subkey, (batch_size, 2), 0, 20, dtype=jnp.int32)
        return key, pairs

    # Condition and body for the while loop
    def cond_fn(state):
        _, x1, y1, pairs = state
        # Check if any pair in the batch is unique
        return jnp.all(jnp.logical_or(pairs[:, 0] == x1, pairs[:, 1] == y1))

    def body_fn(state):
        key, x1, y1, _ = state
        key, pairs = generate_batch(key)
        return key, x1, y1, pairs

    # Initial batch generation
    key, pairs = generate_batch(key)
    initial_state = key, x1, y1, pairs

    # Run the while loop
    _, _, _, unique_pairs = lax.while_loop(cond_fn, body_fn, initial_state)

    # Select the first unique pair from the batch
    x2, y2 = unique_pairs[0]

    return x1, y1, x2, y2

def get_latest_checkpoint_dir(checkpoint_dir):
    checkpoint_paths = sorted(Path(checkpoint_dir).glob('**/checkpoint'), key=os.path.getmtime, reverse=True)
    if checkpoint_paths:
        print("checkpoint found")
        return checkpoint_paths[0].parent
    else:
        print("no checkpoints found")
        return None

# utils.py

def take_damage(
    state: GameState,
    source_id: int,
    target_id: int,
    damage: float,
    damage_type: DamageType
) -> GameState:
    """Apply damage to target unit considering resistances.
    
    Args:
        state: Current game state
        source_id: ID of attacking unit
        target_id: ID of defending unit
        damage: Amount of damage to apply
        damage_type: Type of damage being dealt
        
    Returns:
        Updated game state
    """
    # Handle barrier first
    damage_to_barrier = jnp.minimum(damage, state.units.barrier[target_id, 0])
    remaining_damage = jnp.maximum(jnp.float32(0), damage - damage_to_barrier)
    
    # Reduce barrier
    new_barrier = state.units.barrier.at[target_id, 0].set(
        jnp.maximum(0, state.units.barrier[target_id, 0] - damage_to_barrier)
    )
    
    def process_physical_damage(state):
        def handle_immune(_):
            return state, jnp.float32(0)
            
        def handle_damage(_):
            # Get defense values
            block = state.units.physical_defence[target_id, 0]  # block
            resist = state.units.physical_defence[target_id, 1]  # resist
            
            # Calculate damage
            blocked_damage = jnp.maximum(0, damage - block)
            final_damage = blocked_damage * (1 - resist)
            
            # Update health
            new_health = state.units.health.at[target_id, 0].set(
                jnp.maximum(0, state.units.health[target_id, 0] - final_damage)
            )
            new_state = state.replace(
                units=state.units.replace(health=new_health)
            )
            return new_state, final_damage
            
        # Check immunity
        return lax.cond(
            state.units.physical_defence[target_id, 2],  # immunity
            handle_immune,
            handle_damage,
            None
        )

    def process_magical_damage(state):
        def handle_immune(_):
            return state, jnp.float32(0)
            
        def handle_damage(_):
            block = state.units.magical_defence[target_id, 0]
            resist = state.units.magical_defence[target_id, 1]
            
            blocked_damage = jnp.maximum(0, damage - block)
            final_damage = blocked_damage * (1 - resist)
            
            new_health = state.units.health.at[target_id, 0].set(
                jnp.maximum(0, state.units.health[target_id, 0] - final_damage)
            )
            new_state = state.replace(
                units=state.units.replace(health=new_health)
            )
            return new_state, final_damage

        return lax.cond(
            state.units.magical_defence[target_id, 2],
            handle_immune,
            handle_damage,
            None
        )

    def process_pure_damage(state):
        # Pure damage ignores all defenses
        new_health = state.units.health.at[target_id, 0].set(
            jnp.maximum(0, state.units.health[target_id, 0] - damage)
        )
        new_state = state.replace(
            units=state.units.replace(health=new_health)
        )
        return new_state, damage

    # Update barrier first
    state = state.replace(units=state.units.replace(barrier=new_barrier))

    # Process remaining damage based on type
    return lax.switch(
        damage_type.value - 1,
        [
            lambda x: process_physical_damage(x),
            lambda x: process_magical_damage(x),
            lambda x: process_pure_damage(x)
        ],
        state
    )

def do_damage(
    state: GameState,
    source_id: int,
    target_id: int,
    damage: float,
    damage_type: DamageType,
    return_damage: bool = False
) -> GameState:
    """Handle damage application including lifesteal and return damage.
    
    Args:
        state: Current game state
        source_id: ID of attacking unit
        target_id: ID of defending unit 
        damage: Amount of damage to apply
        damage_type: Type of damage being dealt
        return_damage: Whether to apply return damage
        
    Returns:
        Updated game state
    """
    # Apply initial damage
    new_state, damage_dealt = take_damage(state, source_id, target_id, damage, damage_type)
    
    # Apply lifesteal
    def apply_lifesteal(state, damage_dealt, lifesteal):
        heal_amount = damage_dealt * lifesteal
        new_health = state.units.health.at[source_id, 0].set(
            jnp.minimum(
                state.units.health[source_id, 1],  # max health
                state.units.health[source_id, 0] + heal_amount
            )
        )
        return state.replace(units=state.units.replace(health=new_health))
    
    # Get lifesteal value based on damage type
    lifesteal_amount = lax.switch(
        damage_type.value - 1,
        [
            lambda: state.units.lifesteal[source_id, 0],  # physical
            lambda: state.units.lifesteal[source_id, 1],  # magical
            lambda: state.units.lifesteal[source_id, 2]   # pure
        ]
    )
    
    new_state = apply_lifesteal(new_state, damage_dealt, lifesteal_amount)
    
    # Calculate return damage if needed
    def calc_return_damage(state):
        def get_return_damage():
            base_return = lax.switch(
                damage_type.value - 1,
                [
                    lambda: state.units.physical_defence[target_id, 4],   # physical return
                    lambda: state.units.magical_defence[target_id, 4],    # magical return
                    lambda: state.units.pure_defence[target_id, 4]        # pure return
                ]
            )
            return_rate = lax.switch(
                damage_type.value - 1,
                [
                    lambda: state.units.physical_defence[target_id, 5],   # physical return rate
                    lambda: state.units.magical_defence[target_id, 5],    # magical return rate
                    lambda: state.units.pure_defence[target_id, 5]        # pure return rate
                ]
            )
            return base_return + (damage_dealt * return_rate)
            
        return_amount = get_return_damage()
        return take_damage(state, target_id, source_id, return_amount, damage_type)[0]

    return lax.cond(
        jnp.logical_and(return_damage, damage_dealt > 0),
        calc_return_damage,
        lambda x: x,
        new_state
    )

def do_attack(
    state: GameState,
    source_id: int,
    target_id: int,
    attack_type: AttackType,
    damage_type: DamageType,
    return_damage: bool = False
) -> GameState:
    """Execute an attack based on type.
    
    Args:
        state: Current game state
        source_id: ID of attacking unit
        target_id: ID of defending unit
        attack_type: Type of attack (melee/ranged)
        damage_type: Type of damage
        return_damage: Whether to apply return damage
        
    Returns:
        Updated game state
    """
    # Get base damage based on attack type
    base_damage = lax.switch(
        attack_type.value - 1,
        [
            lambda: state.units.melee_attack[source_id, 0],   # melee base damage
            lambda: state.units.ranged_attack[source_id, 0]   # ranged base damage
        ]
    )
    
    # Apply damage amplification
    damage = base_damage * state.units.damage_amplification[source_id, damage_type.value - 1]
    
    return do_damage(state, source_id, target_id, damage, damage_type, return_damage)

def do_invalid_move(state: GameState, source_id: int, target_id: int) -> GameState:
    """Handle invalid move attempt.
    
    Args:
        state: Current game state
        source_id: ID of unit attempting move
        target_id: ID of target (unused)
        
    Returns:
        Updated game state with incremented step counter
    """
    return state.replace(steps=state.steps + 1)