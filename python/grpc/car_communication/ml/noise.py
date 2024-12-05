import numpy as np


class OUNoise:
    def __init__(
        self,
        action_dim: int,
        std_deviation: float = 0.2,
        theta: float = 0.15,
        dt: float = 1e-2,
        x_initial=None,
    ) -> None:
        self.theta = theta
        self.mean = np.zeros(action_dim)
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def sample(self) -> np.ndarray:
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev, makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self) -> None:
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
