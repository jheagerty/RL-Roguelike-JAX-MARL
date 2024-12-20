# Configuration for the environment
env_config = {
    "BOARD_SIZE": 20,
    "MAX_HEALTH": 100,
    "MELEE_DAMAGE": 40,
    "RANGED_DAMAGE": 20,
    "MELEE_RANGE": 1.6,
    "RANGED_RANGE": 5.1,
    "MOVEMENT_POINTS": 6,
    "ACTION_POINTS": 1,
    "MAX_STEPS": 512,
    "HEROES_PER_TEAM": 1,
    "ABILITIES_PER_HERO": 2,
    "ABILITY_POOL_SIZE": 3,
    }

# Configuration dictionary for the training
train_config = {
    "LR": 5e-4,                           # Learning rate
    "NUM_ENVS": 128,                     # Number of environments
    "NUM_STEPS": 128,                     # Number of steps per environment
    "TOTAL_TIMESTEPS": 1e5,              # Total number of timesteps
    "UPDATE_EPOCHS": 4,                   # Number of update epochs
    "NUM_MINIBATCHES": 4,                 # Number of minibatches
    "GAMMA": 0.99,                        # Discount factor for rewards
    "GAE_LAMBDA": 0.95,                   # Lambda for Generalized Advantage Estimation
    "CLIP_EPS": 0.2,                      # Clipping epsilon for PPO
    "SCALE_CLIP_EPS": False,
    "ENT_COEF": 0.01,                     # Coefficient for entropy term in the loss calculation
    "VF_COEF": 0.5,                       # Coefficient for value function in the loss calculation
    "MAX_GRAD_NORM": 0.5,                 # Maximum gradient norm for gradient clipping
    "ACTIVATION": "relu",                 # Activation function to use in neural networks
    "ENV_NAME": "RL-Roguelike-JAXMARL",   # Name of the environment
    "ANNEAL_LR": True,                    # Flag to indicate if learning rate should be annealed
    "SEED": 30,
    "ENV_KWARGS": {},
    "NUM_SEEDS": 2,
    # WandB Params
    "WANDB_MODE": "disabled",
    "ENTITY": "",
    "PROJECT": "RL-Roguelike",
}

config = {
    "ENV_CONFIG": env_config,
    "TRAIN_CONFIG": train_config,
}