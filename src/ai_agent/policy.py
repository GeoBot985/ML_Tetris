from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from tetris.api import API_COMMANDS

from .environment import snapshot_to_observation


API_ACTIONS = (
    "left",
    "right",
    "soft_drop",
    "hard_drop",
    "rotate_cw",
    "rotate_ccw",
    "hold",
    "pause",
    "restart",
    "start",
    "quit",
)

if set(API_ACTIONS) != API_COMMANDS:
    raise RuntimeError("API_ACTIONS must cover the API command set exactly")

CONTROL_ACTIONS = frozenset({"pause", "restart", "start", "quit"})
PLAY_ACTIONS = tuple(action for action in API_ACTIONS if action not in CONTROL_ACTIONS)
PLAY_ACTION_INDICES = tuple(API_ACTIONS.index(action) for action in PLAY_ACTIONS)
CONTROL_ACTION_INDICES = tuple(API_ACTIONS.index(action) for action in CONTROL_ACTIONS)


def mask_non_play_logits(logits: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    masked[..., list(CONTROL_ACTION_INDICES)] = -1.0e9
    return masked


@dataclass(frozen=True)
class PolicyDecision:
    action_index: int
    action: str
    logits: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


class PPOAgentModel(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int = len(API_ACTIONS),
        hidden_dims: Sequence[int] = (256, 256, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        layers: list[nn.Module] = []
        in_features = observation_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.actor_head = nn.Linear(in_features, action_dim)
        self.critic_head = nn.Linear(in_features, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        features = self.encoder(observations.float())
        logits = self.actor_head(features)
        values = self.critic_head(features).squeeze(-1)
        return logits, values

    def distribution(self, observations: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits, values = self.forward(observations)
        return Categorical(logits=mask_non_play_logits(logits)), values


class PPOPolicy:
    def __init__(
        self,
        model: PPOAgentModel,
        device: str | torch.device | None = None,
        fallback_action_fn: Callable[[dict], str] | None = None,
    ):
        self.model = model
        self.device = torch.device(device or "cpu")
        self.fallback_action_fn = fallback_action_fn
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_observation_dim(cls, observation_dim: int, **model_kwargs) -> "PPOPolicy":
        return cls(PPOAgentModel(observation_dim, **model_kwargs))

    @classmethod
    def from_snapshot(cls, snapshot: dict, **model_kwargs) -> "PPOPolicy":
        observation_dim = snapshot_to_observation(snapshot).shape[0]
        return cls.from_observation_dim(observation_dim, **model_kwargs)

    def act(self, observation, deterministic: bool = False) -> PolicyDecision:
        return self.predict(observation, deterministic=deterministic)

    def predict(self, observation, deterministic: bool = False) -> PolicyDecision:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        dist, values = self.model.distribution(observation_tensor)
        if deterministic:
            action_index = torch.argmax(dist.logits, dim=-1)
        else:
            action_index = dist.sample()
        action_index_int = int(action_index.item())
        action = API_ACTIONS[action_index_int]
        log_prob = dist.log_prob(action_index)
        entropy = dist.entropy()
        value = values.squeeze(0) if values.dim() > 0 else values
        return PolicyDecision(
            action_index=action_index_int,
            action=action,
            logits=dist.logits,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
        )

    def predict_from_snapshot(self, snapshot: dict, deterministic: bool = False) -> PolicyDecision:
        observation = snapshot_to_observation(snapshot)
        return self.predict(observation, deterministic=deterministic)

    def act_from_snapshot(self, snapshot: dict, deterministic: bool = False) -> PolicyDecision:
        return self.predict_from_snapshot(snapshot, deterministic=deterministic)
