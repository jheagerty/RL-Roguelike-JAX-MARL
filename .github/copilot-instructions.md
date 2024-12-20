# Context

We are coding a reinforcement learning project using JAX.

# JAX Compatibility Requirements

All code must be JAX-jittable unless explicitly specified otherwise. This means:
- Use `lax.cond` instead of `if` statements
- Use `lax.switch` instead of match/case or multiple conditionals
- Avoid standard Python loops - use `lax.scan`, `jax.vmap`, or other JAX-compatible alternatives
- Ensure all operations are traceable and differentiable

# Function Signature Requirements

Please include complete type annotations in function definitions, such as:
- Input parameter types
- Return value types using `Tuple` for multiple returns
- JAX array types using `chex.Array`
- State/dataclass types where applicable

# Documentation Requirements

Every function must include a docstring that specifies:
- Function's purpose and effect on game state
- All arguments with their types and meaning
- Return value components
- Any important assumptions about input shapes or values

Code sections must be preceded by comments that:
- Explain what the next operations accomplish
- Highlight any JAX-specific considerations
- Document any game logic or state transitions

Comments should explain both the "what" and "why" of complex operations, particularly for:
- JAX control flow constructs
- State updates and transitions
- Reward calculations
- Terminal state checks

# For example

@partial(jax.jit, static_argnums=[0])
def step_env(
    self,
    key: chex.PRNGKey, 
    state: GameState,
    actions: Dict
) -> Tuple[chex.Array, GameState, Dict, Dict, Dict]:
    """Steps the environment forward by one timestep.
    
    Args:
        key: PRNG key for stochastic transitions
        state: Current game state containing player and enemy information
        actions: Dictionary mapping agent IDs to their chosen actions
        
    Returns:
        Tuple containing:
        - Observation array for next state
        - Updated game state
        - Reward dictionary for each agent
        - Done dictionary indicating episode termination
        - Info dictionary for debugging/logging
    """
    # Convert action dict to array format for processing
    actions = jnp.array([actions[i] for i in self.agents])
    
    # Get current player's action and update game state accordingly
    active_player_idx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
    current_action = actions.at[active_player_idx].get()[0]
    
    # Update state based on action using JAX-compatible control flow
    key, subkey = jax.random.split(key)
    new_state = lax.cond(
        jnp.equal(active_player_idx, 0),
        lambda _: self.function_mapper(subkey, state, current_action, state.player, state.enemy),
        lambda _: self.function_mapper(subkey, state, current_action, state.enemy, state.player),
        operand=None
    )
    ...