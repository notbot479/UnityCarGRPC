

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MEMORY_FRACTION = 0.20
