from torch.functional import Tensor
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
import torch

from typing import Any
import numpy as np
import os

from .model_selector import get_best_model_path
from .buffer import ReplayBuffer
from .critic import CriticModel
from .actor import ActorModel


class DDPGAgent:
    _model_and_object = (
        ('actor','actor_network'),
        ('actor_target','target_actor_network'),
        ('critic','critic_network'),
        ('critic_target','target_critic_network'),
    )

    def __init__(
        self,
        gamma: float = 0.99,
        tau:float = 0.005,
        actor_lr:float = 0.0001,
        critic_lr:float = 0.001,
        target_update_interval:int = 10,
        reply_buffer_capacity:int = 10000,
        # load from dir or best
        load_from_dir: str | None = None,
        load_best_from_dir: str | None = None,
    ) -> None: 
        self._step = 1
        self._critic_loss = float('inf')
        self._actor_loss = float('inf')
        # init parameters
        self.tau = tau
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        # load device and reply buffer
        self.device = torch.device(self._device) 
        self.reply_buffer = ReplayBuffer(capacity=reply_buffer_capacity)  
        # init networks; load trained networks if dir passed
        self._init_models()
        if load_from_dir:
            self.load_model(dir_path=load_from_dir)
        elif load_best_from_dir: 
            self.load_best_model(load_best_from_dir)
        # init optimizers for networks
        self.actor_optimizer = Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic_network.parameters(), lr=critic_lr)

    def save_model(self, dir_path: str, *, ext:str='pth') -> None:
        os.makedirs(dir_path, exist_ok=True)
        for name, obj_name in self._model_and_object:
            model_name = f'{name}.{ext}'
            model_path = os.path.join(dir_path, model_name)
            # get network and save state
            model: nn.Module = self.__getattribute__(obj_name)
            self._save_model(model, model_path)
        # show logs
        name = os.path.basename(dir_path)
        print(f'[DDPG] Save model: {name}')

    def load_best_model(self, dir_path:str) -> None:
        best_model_path = get_best_model_path(dir_path=dir_path)
        if not(best_model_path): return
        self.load_model(dir_path=best_model_path)

    def load_model(self, dir_path:str, *, ext:str='pth') -> None:
        if not(os.path.exists): return
        for name, obj_name in self._model_and_object:
            model_name = f'{name}.{ext}'
            model_path = os.path.join(dir_path, model_name)
            # load networks state from file
            model: nn.Module = self.__getattribute__(obj_name)
            model.load_state_dict(torch.load(model_path))
        # show logs
        name = os.path.basename(dir_path)
        print(f'[DDPG] Model was loaded: {name}')

    @property
    def stats(self) -> dict:
        stats = {
            'actor_loss': self.actor_loss,
            'critic_loss': self.critic_loss,
        }
        return stats

    def show_stats(self) -> None:
        cls_name = self.__class__.__name__
        print(f'{cls_name} stats:')
        for k,v in self.stats.items():
            print(f'- {k.title()}: {v}')

    @property
    def critic_loss(self) -> float:
        return self._critic_loss

    @property
    def actor_loss(self) -> float:
        return self._actor_loss

    def extract_qs(self, outputs: Tensor) -> np.ndarray:
        outputs = outputs.cpu()
        qs = outputs.detach().numpy()[0]
        return qs

    def get_qs(self, inputs: dict[str, Any]) -> np.ndarray:
        '''get prediction from actor model'''
        inputs = self.extract_inputs([inputs,])
        tensor = self.actor_network(**inputs).to(self.device)
        qs = self.extract_qs(tensor)
        return qs

    def train(self, *, terminal_state:bool=False, batch_size:int = 64) -> None:
        if terminal_state: self._step += 1
        if not(self.reply_buffer.ready): return 
        print(f'[DDPG] Train model. BatchSize: {batch_size}')
        self._train(batch_size=batch_size)

    def train_on_episode_end(
        self, 
        batch_size: int = 64, 
        batches_count: int = 50,
    ) -> None:
        self._step += 1
        if not(self.reply_buffer.ready): return 
        print(f'[DDPG] Train on episode end. Batches: {batches_count}, BatchSize: {batch_size}')
        [self.train(batch_size=batch_size) for _ in range(batches_count)]

    def extract_inputs(self, data: list[dict[str,Any]]) -> dict[str, Tensor]:
        inputs = {}
        [{inputs.setdefault(k, []).append(v) for k,v in row.items()} for row in data]
        for k, v in inputs.items(): inputs[k] = self.convert_to_tensor_and_stack(v)
        return inputs

    def convert_to_tensor_and_stack(self, data:list[Any]) -> Tensor:
        tensors = [self.convert_to_tensor(i) for i in data]
        tensor = torch.stack(tensors, dim=0)
        return tensor

    def convert_to_tensor(self, data: Any) -> Tensor:
        np_data = np.array(data, dtype=np.float32).T
        tensor = torch.tensor(np_data, dtype=torch.float32).to(self.device)
        if not(tensor.dim()): tensor = tensor.unsqueeze(0)
        return tensor

    @property
    def step(self) -> int:
        return self._step

    @property
    def cuda(self) -> bool:
        return torch.cuda.is_available()

    def _init_models(self) -> None:
        # init networks
        self.actor_network = ActorModel().to(self.device)
        self.critic_network = CriticModel().to(self.device)
        # init target networks
        self.target_actor_network = ActorModel().to(self.device)
        self.target_critic_network = CriticModel().to(self.device)
        # hard update target networks weights from networks
        self._hard_update_target_networks()

    def _save_model(self, model: nn.Module, path:str) -> None:
        torch.save(model.state_dict(),path)

    @property
    def _device(self) -> str:
        device = 'cuda' if self.cuda else 'cpu'
        return device

    def _train(self, batch_size) -> None:
        sample = self.reply_buffer.sample(batch_size=batch_size)
        if not(sample): return
        # convert batch items to tensors
        states, actions, rewards, next_states, dones = sample
        states = self.extract_inputs(states)
        actions = self.convert_to_tensor_and_stack(actions)
        rewards = self.convert_to_tensor_and_stack(rewards)
        next_states = self.extract_inputs(next_states)
        dones = self.convert_to_tensor_and_stack(dones)
        # update critic network
        critic_loss = self._calculate_critic_loss(
            states = states,
            rewards = rewards,
            dones = dones,
            actions = actions,
            next_states = next_states,
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # update actor network
        actor_loss = self._calculate_actor_loss(states=states) 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # update target networks based on counter
        _b = self.step % self.target_update_interval == 0
        if self.target_update_interval > 0 and _b:
            self._soft_update_target_networks()

    def _calculate_critic_loss(
        self, 
        states: dict[str,Tensor], 
        rewards: Tensor, 
        dones: Tensor, 
        actions: Tensor, 
        next_states: dict[str,Tensor],
    ):
        next_action = self.target_actor_network(**next_states)
        target_Q_next = self.target_critic_network(
            **next_states, 
            actor_action=next_action,
        ) 
        target_Q = rewards + (self.gamma * target_Q_next * (1 - dones))
        current_Q = self.critic_network(**states, actor_action=actions)
        critic_loss = F.mse_loss(current_Q, target_Q)
        # save critic loss to variable
        self._critic_loss = float(critic_loss)
        return critic_loss

    def _calculate_actor_loss(self, states: dict[str,Tensor]):
        actor_predict = self.actor_network(**states)
        critic_predict = self.critic_network(**states, actor_action=actor_predict)
        actor_loss = -critic_predict.mean()
        # save actor loss to variable
        self._actor_loss = float(actor_loss)
        return actor_loss
    
    def _hard_update_target_networks(self):
        a = self.actor_network.state_dict()
        c = self.critic_network.state_dict()
        # update by copy weights
        self.target_actor_network.load_state_dict(a)
        self.target_critic_network.load_state_dict(c)
    
    def _soft_update_target_networks(self):
        # soft update for actor
        ta = self.target_actor_network.parameters()
        a = self.actor_network.parameters()
        for tp, p in zip(ta,a):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        # soft update for ctitic
        tc = self.target_critic_network.parameters()
        c = self.critic_network.parameters()
        for tp, p in zip(tc,c):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data) 
