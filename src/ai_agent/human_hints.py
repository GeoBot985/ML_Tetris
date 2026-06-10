from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .environment import snapshot_to_observation
from .policy import API_ACTIONS


HUMAN_HINT_ACTIONS = {"left", "right", "soft_drop", "hard_drop", "rotate_cw", "rotate_ccw", "hold"}


def write_human_hint(path: Path, snapshot: dict[str, Any], action: str, *, difficulty: str | None = None) -> bool:
    if action not in HUMAN_HINT_ACTIONS:
        return False

    observation = snapshot_to_observation(snapshot)
    record = {
        "type": "human_hint",
        "version": 1,
        "timestamp": time.time(),
        "action": action,
        "action_index": API_ACTIONS.index(action),
        "difficulty": difficulty,
        "observation": observation.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return True


def count_human_hints(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "human_hint" and payload.get("action") in HUMAN_HINT_ACTIONS:
            count += 1
    return count


def load_human_hints(path: Path, *, observation_dim: int | None = None) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not path.exists():
        return None

    observations = []
    actions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        action = payload.get("action")
        observation = payload.get("observation")
        if action not in HUMAN_HINT_ACTIONS or not isinstance(observation, list):
            continue
        if observation_dim is not None and len(observation) != observation_dim:
            continue
        observations.append(torch.tensor(observation, dtype=torch.float32))
        actions.append(API_ACTIONS.index(action))

    if not observations:
        return None
    return torch.stack(observations), torch.tensor(actions, dtype=torch.long)
