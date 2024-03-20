from torch.functional import Tensor
import torch.nn.functional as F
from torch.optim import Adam
import torch

from typing import Any
import numpy as np

from .buffer import ReplayBuffer
from .critic import CriticModel
from .actor import ActorModel


class DDPGAgent:
    def __init__(
        self,
        gamma: float = 0.99,
        tau:float = 0.005,
        actor_lr:float = 0.001,
        critic_lr:float = 0.001,
        target_update_interval:int = 10,
        reply_buffer_capacity:int | None = None,
    ) -> None: # TODO load trained networks
        # init parameters
        self._step = 0
        self.tau = tau
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        # load device and reply buffer
        self.device = torch.device(self._device) 
        self.reply_buffer = ReplayBuffer(capacity=reply_buffer_capacity) 
        # init networks
        self.actor_network = ActorModel().to(self.device)
        self.actor_optimizer = Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_network = CriticModel().to(self.device)
        self.critic_optimizer = Adam(self.critic_network.parameters(), lr=critic_lr)
        # init target networks and hard load weights
        self.target_actor_network = ActorModel().to(self.device)
        self.target_critic_network = CriticModel().to(self.device)
        self._hard_update_target_networks()

    def extract_qs(self, outputs: Tensor) -> np.ndarray:
        outputs = outputs.to(self.device) 
        qs = outputs.detach().numpy()[0]
        return qs

    def get_qs(self, inputs: dict[str, Any]) -> np.ndarray:
        '''get prediction from actor model'''
        inputs = self.extract_inputs([inputs,])
        tensor = self.actor_network(**inputs).to(self.device)
        qs = self.extract_qs(tensor)
        return qs

    def predict_action(self, inputs: dict[str, Any]) -> int:
        qs = self.get_qs(inputs=inputs)
        action = np.argmax(qs)
        return int(action)

    def train(self, *, terminal_state:bool=False, batch_size:int = 64) -> None:
        if terminal_state: self._step += 1
        self._train(batch_size=batch_size)

    def train_on_episode_end(
        self, 
        batch_size: int = 64, 
        batches_count: int = 50,
    ) -> None:
        self._step += 1
        [self.train(batch_size=batch_size) for _ in range(batches_count)]

    def extract_inputs(self, data: list[dict[str,Any]]) -> dict[str, Tensor]:
        tensor_inputs = {}
        for sample in data:
            for key, value in sample.items():
                tensor = self.convert_to_tensor(value)
                if not(tensor.dim()): tensor = tensor.unsqueeze(0)
                tensor_inputs.setdefault(key, []).append(tensor)
        for k, v in tensor_inputs.items():
            tensor_inputs[k] = torch.stack(v, dim=0)
        return tensor_inputs

    def convert_to_tensor(self, data: Any) -> Tensor:
        np_data = np.array(data, dtype=np.float32).T
        tensor = torch.tensor(np_data, dtype=torch.float32).to(self.device)
        return tensor

    @property
    def step(self) -> int:
        return self._step

    @property
    def cuda(self) -> bool:
        return torch.cuda.is_available()
    

    @property
    def _device(self) -> str:
        device = 'gpu' if self.cuda else 'cpu'
        return device

    def _train(self, batch_size) -> None:
        sample = self.reply_buffer.sample(batch_size=batch_size)
        tensors = map(self.convert_to_tensor, sample)
        (states, actions, rewards, next_states, dones) = tensors
        # Update Critic
        critic_loss = self._get_critic_loss(
            states = states,
            rewards = rewards,
            dones = dones,
            actions = actions,
            next_states = next_states,
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Update Actor
        actor_loss = self._get_actor_loss(states=states) 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Update target networks
        _b = self._step % self.target_update_interval == 0
        if self.target_update_interval > 0 and _b:
            self._soft_update_target_networks()

    def _get_critic_loss(
        self, 
        states: Tensor, 
        rewards: Tensor, 
        dones: Tensor, 
        actions: Tensor, 
        next_states: Tensor,
    ):
        next_action = self.target_actor_network(next_states)
        target_Q = self.target_critic_network(next_states, next_action) 
        target_Q = rewards + (1 - dones) * self.gamma * target_Q.detach()
        current_Q = self.critic_network(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)
        return critic_loss

    def _get_actor_loss(self, states: Tensor):
        actor_predict = self.actor_network(states)
        actor_loss = -self.critic_network(states, actor_predict).mean()
        return actor_loss
    
    def _hard_update_target_networks(self):
        a = self.actor_network.state_dict()
        c = self.critic_network.state_dict()
        # update by copy weights
        self.target_actor_network.load_state_dict(a)
        self.target_critic_network.load_state_dict(c)
    
    def _soft_update_target_networks(self):
        # soft updat for actor
        ta = self.target_actor_network.parameters()
        a = self.actor_network.parameters()
        for tp, p in zip(ta,a):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        # soft update for ctitic
        tc = self.target_critic_network.parameters()
        c = self.critic_network.parameters()
        for tp, p in zip(tc,c):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data) 
