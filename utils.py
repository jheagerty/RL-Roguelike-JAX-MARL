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

def take_damage(attacker, defender, damage, damage_type: DamageType):
    """Apply damage to defender considering barriers first, then resistances"""
    
    def apply_barrier_and_damage(inputs):
        attacker, defender, damage = inputs
        
        # Calculate how much damage the barrier absorbs
        damage_to_barrier = jnp.minimum(defender.barrier_current, damage)
        remaining_damage = damage - damage_to_barrier
        
        # Update barrier
        new_barrier_current = jnp.maximum(0, defender.barrier_current - damage_to_barrier)
        defender = defender.replace(
            barrier_current=new_barrier_current,
            barrier_percentage=new_barrier_current / defender.barrier_max
        )
        
        # If there's remaining damage, apply it considering resistances
        def process_physical_damage(inputs):
            attacker, defender, damage = inputs
            
            def handle_immune(_):
                return attacker, defender, jnp.float32(damage_to_barrier)
                
            def handle_damage(_):
                # Calculate damage with blocks and resists
                blocked_damage = jnp.maximum(0, damage - defender.physical_block)
                final_damage = blocked_damage * (1 - defender.physical_resist)
                
                new_defender = defender.replace(
                    health_current=jnp.float32(jnp.maximum(0, defender.health_current - final_damage))
                )
                return attacker, new_defender, jnp.float32(damage_to_barrier + final_damage)
                
            return lax.cond(
                defender.physical_immunity,
                handle_immune,
                handle_damage,
                None
            )

        def process_magical_damage(inputs):
            attacker, defender, damage = inputs
            
            def handle_immune(_):
                return attacker, defender, jnp.float32(damage_to_barrier)
                
            def handle_damage(_):
                blocked_damage = jnp.maximum(0, damage - defender.magical_block)
                final_damage = blocked_damage * (1 - defender.magical_resist)
                
                new_defender = defender.replace(
                    health_current=jnp.float32(jnp.maximum(0, defender.health_current - final_damage))
                )
                return attacker, new_defender, jnp.float32(damage_to_barrier + final_damage)

            return lax.cond(
                defender.magical_immunity,
                handle_immune,
                handle_damage,
                None
            )

        def process_pure_damage(inputs):
            attacker, defender, damage = inputs
            new_defender = defender.replace(
                health_current=jnp.float32(jnp.maximum(0, defender.health_current - damage))
            )
            return attacker, new_defender, jnp.float32(damage_to_barrier + damage)

        # Only process remaining damage if there is any
        def process_remaining(_):
            return lax.switch(
                damage_type.value - 1,
                [
                    lambda x: process_physical_damage(x),
                    lambda x: process_magical_damage(x),
                    lambda x: process_pure_damage(x)
                ],
                (attacker, defender, remaining_damage)
            )
            
        def no_remaining(_):
            return attacker, defender, jnp.float32(damage_to_barrier)
            
        return lax.cond(
            remaining_damage > 0,
            process_remaining,
            no_remaining,
            None
        )

    return apply_barrier_and_damage((attacker, defender, damage))

def do_damage(attacker, defender, damage, damage_type: DamageType, return_damage: bool = False): # TODO: check how we do with negative damage
    """Handle damage application, lifesteal, and return damage mechanics"""
    # Apply initial damage
    attacker, defender, damage_dealt = take_damage(attacker, defender, damage, damage_type) #TODO: if defender is immune, do no damage at all? as in, such that nothing else gets triggered
    
    # Apply lifesteal based on damage type
    def apply_lifesteal(inputs):
        attacker, damage_dealt, lifesteal = inputs
        heal_amount = damage_dealt * lifesteal
        new_health = jnp.minimum(attacker.health_max, attacker.health_current + heal_amount)
        return attacker.replace(health_current=new_health)
    
    lifesteal_amount = lax.switch(
        damage_type.value - 1,
        [
            lambda x: x.physical_lifesteal,
            lambda x: x.magical_lifesteal,
            lambda x: x.pure_lifesteal
        ],
        attacker
    )
    
    attacker = apply_lifesteal((attacker, damage_dealt, lifesteal_amount))
    
    # Calculate return damage
    def calc_physical_return(damage_dealt):
        return defender.physical_damage_return + (damage_dealt * defender.physical_damage_return_rate)
        
    def calc_magical_return(damage_dealt):
        return defender.magical_damage_return + (damage_dealt * defender.magical_damage_return_rate)
        
    def calc_pure_return(damage_dealt):
        return defender.pure_damage_return + (damage_dealt * defender.pure_damage_return_rate)
    
    return_damage_amount = lax.switch(
        damage_type.value - 1,
        [
            lambda x: calc_physical_return(x),
            lambda x: calc_magical_return(x),
            lambda x: calc_pure_return(x)
        ],
        damage_dealt
    )
    
    # Handle return damage if needed
    def apply_return_damage(_):
        new_attacker, _, _ = take_damage(defender, attacker, return_damage_amount, damage_type)
        return new_attacker, defender
        
    def skip_return(_):
        return attacker, defender
    
    return lax.cond(
        jnp.logical_and(
            jnp.logical_not(return_damage),
            return_damage_amount > 0
        ),
        apply_return_damage,
        skip_return,
        None
    )

def do_attack(attacker, defender, attack_type: AttackType, damage_type: DamageType, return_damage: bool = False):
    """Execute an attack based on type"""
    def handle_melee(args):
        attacker, defender = args
        damage = attacker.melee_base_attack_damage
        return do_damage(attacker, defender, damage, damage_type, return_damage)
        
    def handle_ranged(args):
        attacker, defender = args
        damage = attacker.ranged_base_attack_damage
        return do_damage(attacker, defender, damage, damage_type, return_damage)
    
    return lax.switch(
        attack_type.value - 1,  # Convert enum to 0-based index
        [
            handle_melee,
            handle_ranged
        ],
        (attacker, defender)
    )