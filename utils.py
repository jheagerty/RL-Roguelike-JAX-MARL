import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import replace
import os
from pathlib import Path
import chex

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