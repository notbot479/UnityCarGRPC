from torch.optim.lr_scheduler import ExponentialLR
from torch.functional import Tensor
from torch.optim import Adam
import torch.nn as nn
import torch

from typing import Any
import numpy as np
import copy
import os

from .model_selector import get_best_model_path
from .buffer import ReplayBuffer
from .critic import CriticModel
from .actor import ActorModel
from .noise import OUNoise


class DDPGAgent:
    _model_and_object = (
        ("actor", "actor_network"),
        ("actor_optimizer", "actor_optimizer"),
        ("critic", "critic_network"),
        ("critic_optimizer", "critic_optimizer"),
    )
    _default_loss = float("inf")

    def __init__(
        self,
        # base parameters
        action_dim: int = 2,
        max_action: int = 1,
        reply_buffer_capacity: int = 100000,
        # agent parameters
        discount: float = 0.99,
        tau: float = 0.001,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-4,
        lr_decay: float = 0.9995,
        # load from dir or best
        load_from_dir: str | None = None,
        load_best_from_dir: str | None = None,
        # extra settings
        use_mock_image_if_no_cuda: bool = False,
    ) -> None:
        self._step = 1
        self._critic_loss: list[float] = []
        self._actor_loss: list[float] = []
        # init parameters
        self.tau = tau
        self.discount = discount
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.lr_decay = lr_decay
        self.max_action = max_action
        self.action_dim = action_dim
        self.use_mock_image_if_no_cuda = use_mock_image_if_no_cuda
        # load device, reply buffer, noise
        self.device = torch.device(self._device)
        self.reply_buffer = ReplayBuffer(capacity=reply_buffer_capacity)
        self.noise = OUNoise(action_dim=action_dim)
        # init networks; load trained networks if dir passed
        self._init_networks()
        if load_from_dir:
            self.load_model(dir_path=load_from_dir)
        elif load_best_from_dir:
            self.load_best_model(load_best_from_dir)
        # init learning rate scheduler
        if self.lr_decay:
            self._init_lr_schedulers()

    @property
    def step(self) -> int:
        return self._step

    def update_schedulers(self) -> None:
        if not (self.lr_decay and self.reply_buffer.ready):
            return
        self.critic_scheduler.step()

    def save_model(self, dir_path: str, *, ext: str = "pth") -> None:
        os.makedirs(dir_path, exist_ok=True)
        for name, obj_name in self._model_and_object:
            model_name = f"{name}.{ext}"
            model_path = os.path.join(dir_path, model_name)
            # get network and save state
            model: nn.Module = self.__getattribute__(obj_name)
            self._save_model(model, model_path)
        # show logs
        name = os.path.basename(dir_path)
        print(f"[DDPG] Save model: {name}")

    def load_best_model(self, dir_path: str) -> None:
        best_model_path = get_best_model_path(dir_path=dir_path)
        if not (best_model_path):
            return
        self.load_model(dir_path=best_model_path)

    def _load_model(self, obj_name: str, model_path: str) -> None:
        model: nn.Module = self.__getattribute__(obj_name)
        weights = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(weights)

    def load_model(self, dir_path: str, *, ext: str = "pth") -> None:
        if not (os.path.exists):
            return
        for name, obj_name in self._model_and_object:
            model_name = f"{name}.{ext}"
            model_path = os.path.join(dir_path, model_name)
            # load networks state from file
            try:
                self._load_model(obj_name=obj_name, model_path=model_path)
            except Exception as e:
                print(f"Failed load Model({model_name}) by path, reason: {e}")
        # init target networks
        self._init_target_networks()
        # show logs
        name = os.path.basename(dir_path)
        print(f"[DDPG] Model was loaded: {name}")

    @property
    def stats(self) -> dict:
        stats = {
            "actor_loss": self.actor_avg_loss,
            "critic_loss": self.critic_avg_loss,
            "critic_optimizer_lr": self.critic_optimizer_lr,
        }
        return stats

    def show_stats(self) -> None:
        cls_name = self.__class__.__name__
        print(f"{cls_name} stats:")
        for k, v in self.stats.items():
            print(f"- {k.title()}: {v}")

    @property
    def critic_optimizer_lr(self) -> float:
        return self.critic_optimizer.param_groups[0]["lr"]

    @property
    def critic_avg_loss(self) -> float:
        loss = self._critic_loss
        loss = self._get_avg(loss)
        return loss

    @property
    def actor_avg_loss(self) -> float:
        loss = self._actor_loss
        loss = self._get_avg(loss)
        return loss

    @property
    def critic_loss(self) -> float:
        loss = self._critic_loss
        loss = loss[-1] if loss else self._default_loss
        return loss

    @property
    def actor_loss(self) -> float:
        loss = self._actor_loss
        loss = loss[-1] if loss else self._default_loss
        return loss

    def extract_qs(self, outputs: Tensor) -> np.ndarray:
        outputs = outputs.cpu()
        qs = outputs.data.numpy().flatten()
        return qs

    def get_qs(self, inputs: dict[str, Any], exploration: bool = False) -> np.ndarray:
        """get prediction from actor model in eval mode"""
        inputs = self.extract_inputs(
            [
                inputs,
            ]
        )
        tensor = self.actor_network.predict(**inputs)
        qs = self.extract_qs(tensor)
        # add exploration noise and clip out of limits
        if exploration:
            qs += self.noise.sample()
        qs = np.clip(qs, -self.max_action, self.max_action)
        return qs

    def train(
        self,
        terminal_state: bool = False,
        batch_size: int = 64,
        *,
        prefix: str | None = None,
        state_id: int | None = None,
    ) -> None:
        if self.reply_buffer.ready:
            prefix = self._get_prefix(prefix=prefix, state_id=state_id)
            print(f"[DDPG]{prefix}Train model. BatchSize: {batch_size}")
            self._train(batch_size=batch_size)
        if terminal_state:
            self._step += 1

    def train_on_episode_end(
        self,
        batch_size: int = 64,
        batches_count: int = 50,
    ) -> None:
        if self.reply_buffer.ready:
            bc, bs = batches_count, batch_size
            print(f"[DDPG] Train on episode end. Batches: {bc}, BatchSize: {bs}")
            [self.train(prefix=f"[{i}/{bc}]", batch_size=bs) for i in range(1, bc + 1)]
        self._step += 1

    def extract_inputs(self, data: list[dict[str, Any]]) -> dict[str, Tensor]:
        inputs = {}
        [{inputs.setdefault(k, []).append(v) for k, v in row.items()} for row in data]
        for k, v in inputs.items():
            inputs[k] = self.convert_to_tensor_and_stack(v)
        return inputs

    def convert_to_tensor_and_stack(self, data: list[Any]) -> Tensor:
        tensors = [self.convert_to_tensor(i) for i in data]
        tensor = torch.stack(tensors, dim=0)
        tensor = tensor.to(self.device)
        return tensor

    def convert_to_tensor(self, data: Any) -> Tensor:
        np_data = np.array(data, dtype=np.float32).T
        tensor = torch.tensor(np_data, dtype=torch.float32).to(self.device)
        # add batch dim for tensor if singleton
        if not (tensor.dim()):
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor

    @property
    def cuda(self) -> bool:
        return torch.cuda.is_available()

    def reset_loss(self) -> None:
        self._critic_loss.clear()
        self._actor_loss.clear()

    @staticmethod
    def _get_prefix(*, prefix: str | None = None, state_id: int | None = None) -> str:
        if prefix:
            return f"{prefix} "
        elif state_id:
            return f"[{state_id}] "
        return " "

    def _init_lr_schedulers(self) -> None:
        self.critic_scheduler = ExponentialLR(
            self.critic_optimizer,
            gamma=self.lr_decay,
        )

    def _get_avg(self, data: list[float]) -> float:
        if not (data):
            return self._default_loss
        avg = np.array(data).mean()
        return avg

    def _init_target_networks(self) -> None:
        self.target_actor_network = copy.deepcopy(self.actor_network)
        self.target_critic_network = copy.deepcopy(self.critic_network)

    def _init_networks(self) -> None:
        mock_image = self.use_mock_image_if_no_cuda and not (self.cuda)
        # init actor network
        self.actor_network = ActorModel(
            action_dim=self.action_dim,
            max_action=self.max_action,
            mock_image=mock_image,
        ).to(self.device)
        self.actor_optimizer = Adam(
            params=self.actor_network.parameters(),
            lr=self.actor_lr,
        )
        # init critic network
        self.critic_network = CriticModel(
            action_dim=self.action_dim,
            max_action=self.max_action,
            mock_image=mock_image,
        ).to(self.device)
        self.critic_optimizer = Adam(
            params=self.critic_network.parameters(),
            lr=self.critic_lr,
        )
        # init target networks
        self._init_target_networks()

    def _save_model(self, model: nn.Module, path: str) -> None:
        torch.save(model.state_dict(), path)

    @property
    def _device(self) -> str:
        device = "cuda" if self.cuda else "cpu"
        return device

    def _train(self, batch_size) -> None:
        sample = self.reply_buffer.sample(batch_size=batch_size)
        if not (sample):
            return
        # convert batch items to tensors
        states, actions, rewards, next_states, dones = sample
        states = self.extract_inputs(states)
        actions = self.convert_to_tensor_and_stack(actions)
        rewards = self.convert_to_tensor_and_stack(rewards)
        next_states = self.extract_inputs(next_states)
        dones = self.convert_to_tensor_and_stack(dones)
        # update critic network
        critic_loss = self._calculate_critic_loss(
            states=states,
            rewards=rewards,
            dones=dones,
            actions=actions,
            next_states=next_states,
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # update actor network
        actor_loss = self._calculate_actor_loss(states=states)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # update target networks
        self._soft_update_target_networks()

    def _calculate_critic_loss(
        self,
        states: dict[str, Tensor],
        rewards: Tensor,
        dones: Tensor,
        actions: Tensor,
        next_states: dict[str, Tensor],
    ) -> Tensor:
        target_Q = self.target_critic_network(
            **next_states,
            actor_action=self.target_actor_network(**next_states),
        )
        target_Q = rewards + ((1 - dones) * self.discount * target_Q).detach()

        current_Q = self.critic_network(**states, actor_action=actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self._critic_loss.append(float(critic_loss))
        return critic_loss

    def _calculate_actor_loss(self, states: dict[str, Tensor]) -> Tensor:
        critic_predict = self.critic_network(
            **states,
            actor_action=self.actor_network(**states),
        )
        actor_loss = -critic_predict.mean()
        self._actor_loss.append(float(actor_loss))
        return actor_loss

    def _soft_update_target_networks(self) -> None:
        self._soft_update_network(
            network=self.actor_network,
            target_network=self.target_actor_network,
        )
        self._soft_update_network(
            network=self.critic_network,
            target_network=self.target_critic_network,
        )

    def _soft_update_network(
        self,
        network: nn.Module,
        target_network: nn.Module,
    ) -> None:
        for p, tp in zip(network.parameters(), target_network.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
