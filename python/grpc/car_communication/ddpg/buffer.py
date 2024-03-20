from collections import deque, namedtuple
import random

from .parameters import REPLAY_MEMORY_SIZE


Transition = namedtuple(
    'Transition', 
    ['state', 'action', 'reward', 'next_state', 'done'],
)


class ReplayBuffer:
    def __init__(self, capacity: int | None):
        self.capacity = capacity if capacity else REPLAY_MEMORY_SIZE
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
