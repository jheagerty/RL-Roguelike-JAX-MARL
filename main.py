import time
import jax
from config import train_config
from training import make_train

def main():
    # Set a random seed for reproducibility
    rng = jax.random.PRNGKey(42)

    # Create the training function from configuration
    train_jit = make_train(train_config)

    print("Training start")

    # Record the start time
    start = time.time()

    # Execute the training function
    out = train_jit(rng)

    # Calculate and print the elapsed time
    elapsed_time = time.time() - start
    print(f"Training completed in {elapsed_time} seconds")

    return out

if __name__ == "__main__":
    main()
