from typing import NamedTuple
from collections import deque
from typing import Any
import numpy as np
import random


Inputs = dict[str, Any]


class Transition(NamedTuple):
    state: Inputs
    action: np.ndarray
    reward: float
    next_state: Inputs
    done: bool


class BufferSample(NamedTuple):
    states: list[Inputs]
    actions: list[np.ndarray]
    rewards: list[float]
    next_states: list[Inputs]
    dones: list[bool]


class ReplayBuffer:
    def __init__(self, capacity: int = 50000, *, min_capacity: int = 1000):
        self._capacity = capacity
        self._min_capacity = min_capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def min_capacity(self) -> int:
        return self._min_capacity

    @property
    def ready(self) -> bool:
        status = len(self.buffer) >= self._min_capacity
        return status

    def store(
        self,
        state: Inputs,
        action: np.ndarray,
        reward: float,
        next_state: Inputs,
        done: bool,
    ) -> None:
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int = 64) -> BufferSample | None:
        if not (self.ready) or batch_size >= len(self.buffer):
            return
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        buffer_sample = BufferSample(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )
        return buffer_sample


def _test():
    reply_buffer = ReplayBuffer()
    items = 1000
    # fill reply buffer
    for i in range(items):
        state = {"i": i}
        action = np.array(
            [
                0.1,
            ]
            * 5
        )
        reward = random.random()
        next_state = {"i": i + 1}
        done = random.randint(0, 5) == 3
        reply_buffer.store(state, action, reward, next_state, done)
    # test buffer
    print(f"Buffer len: {len(reply_buffer)}")
    sample = reply_buffer.sample()
    if not (sample):
        print(f"Required min capacity: {reply_buffer.min_capacity}")
        return
    states, _, _, _, _ = sample
    print(states)


if __name__ == "__main__":
    _test()
