import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import replace
import os
from pathlib import Path
import chex
from data_classes import AttackType, DamageType

# Util Functions
def euclidean_distance(x1, y1, x2, y2):
    return jnp.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def is_within_bounds(x, y, max_bound=20):
    return jnp.logical_and(jnp.logical_and(0 <= x, x < max_bound),
                           jnp.logical_and(0 <= y, y < max_bound))

def is_collision(x1, y1, x2, y2):
    return jnp.logical_and(x1 == x2, y1 == y2)

def do_invalid_move(state, unit, target):
    return replace(
        state,
        steps=jnp.int32(state.steps + 1)
    )

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
    
def get_available_actions(
        unit,
        target,
        state,
        action_functions,
        ) -> chex.Array:
    return jnp.array([action.is_valid(state, unit, target) for action in action_functions], dtype=jnp.int32)

def take_damage(attacker, defender, damage, damage_type: DamageType, return_damage: bool = False):
    """Apply damage to defender considering resistances and return damage mechanics"""
    def process_physical_damage(inputs):
        attacker, defender, damage = inputs
        
        def handle_immune(_):
            # Add damage=0 and return_dmg=0 to match handle_damage output type
            return attacker, defender, jnp.float32(0), jnp.float32(0)
            
        def handle_damage(_):
            # Calculate damage
            blocked_damage = jnp.maximum(0, damage - defender.physical_block)
            final_damage = blocked_damage * (1 - defender.physical_resist)
            return_dmg = defender.physical_damage_return + (final_damage * defender.physical_damage_return_rate)
            
            new_defender = defender.replace(
                health_current=jnp.float32(jnp.maximum(0, defender.health_current - final_damage))
            )
            return attacker, new_defender, jnp.float32(final_damage), jnp.float32(return_dmg)
            
        return lax.cond(
            defender.physical_immunity,
            handle_immune,
            handle_damage,
            None
        )

    def process_magical_damage(inputs):
        attacker, defender, damage = inputs
        
        def handle_immune(_):
            return attacker, defender, jnp.float32(0), jnp.float32(0)
            
        def handle_damage(_):
            blocked_damage = jnp.maximum(0, damage - defender.magical_block)
            final_damage = blocked_damage * (1 - defender.magical_resist)
            return_dmg = defender.magical_damage_return + (final_damage * defender.magical_damage_return_rate)
            
            new_defender = defender.replace(
                health_current=jnp.float32(jnp.maximum(0, defender.health_current - final_damage))
            )
            return attacker, new_defender, jnp.float32(final_damage), jnp.float32(return_dmg)

        return lax.cond(
            defender.magical_immunity,
            handle_immune,
            handle_damage,
            None
        )

    def process_pure_damage(inputs):
        attacker, defender, damage = inputs
        return_dmg = defender.pure_damage_return + (damage * defender.pure_damage_return_rate)
        
        new_defender = defender.replace(
            health_current=jnp.float32(jnp.maximum(0, defender.health_current - damage))
        )
        return attacker, new_defender, jnp.float32(damage), jnp.float32(return_dmg)

    # Process initial damage based on type
    attacker, defender, damage_dealt, return_damage_amount = lax.switch(
        damage_type.value - 1,
        [
            lambda x: process_physical_damage(x),
            lambda x: process_magical_damage(x),
            lambda x: process_pure_damage(x)
        ],
        (attacker, defender, damage)
    )
    
    # Handle return damage if needed
    def apply_return_damage(_):
        new_attacker, _, _, _ = lax.switch(
            damage_type.value - 1,
            [
                lambda x: process_physical_damage(x),
                lambda x: process_magical_damage(x),
                lambda x: process_pure_damage(x)
            ],
            (defender, attacker, return_damage_amount)
        )
        return new_attacker, defender
        
    def skip_return(_):
        return attacker, defender
    
    # Only apply return damage if not already processing return damage
    return lax.cond(
        jnp.logical_and(
            jnp.logical_not(return_damage),
            return_damage_amount > 0
        ),
        apply_return_damage,
        skip_return,
        None
    )

def do_damage(attacker, defender, damage, damage_type: DamageType, return_damage: bool = False):
    """Handle damage application"""
    return take_damage(attacker, defender, damage, damage_type, return_damage)

def do_attack(attacker, defender, attack_type: AttackType, damage_type: DamageType, return_damage: bool = False):
    """Execute an attack based on type"""
    match attack_type:
        case AttackType.MELEE:
            damage = attacker.melee_base_attack_damage
            return do_damage(attacker, defender, damage, damage_type, return_damage)
        case AttackType.RANGED:
            damage = attacker.ranged_base_attack_damage
            return do_damage(attacker, defender, damage, damage_type, return_damage)