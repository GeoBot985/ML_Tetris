from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory as mp_shared_memory
from typing import Callable

import numpy as np

from .environment import ObservationLayout, TetrisEnvironment, build_observation_layout, snapshot_to_observation
from .policy import API_ACTIONS
from tetris.piece_source import classic_uniform_source

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("gymnasium is required for the vectorized environment backend") from exc

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("stable-baselines3 is required for the vectorized environment backend") from exc


@dataclass
class SharedObservationBuffer:
    shared_memory: mp_shared_memory.SharedMemory
    array: np.ndarray

    def close(self) -> None:
        self.shared_memory.close()

    def unlink(self) -> None:
        self.shared_memory.unlink()


def create_shared_observation_buffer(layout: ObservationLayout, *, name: str | None = None) -> SharedObservationBuffer:
    shm = mp_shared_memory.SharedMemory(create=True, size=layout.size * np.dtype(np.float32).itemsize, name=name)
    array = np.ndarray((layout.size,), dtype=np.float32, buffer=shm.buf)
    array.fill(0.0)
    return SharedObservationBuffer(shared_memory=shm, array=array)


class TetrisGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        *,
        difficulty,
        seed: int | None = None,
        queue_size: int = 5,
        use_shared_memory: bool = False,
        piece_source: str = "classic_uniform",
    ):
        super().__init__()
        self.piece_source = piece_source
        self.core = TetrisEnvironment(
            difficulty=difficulty,
            queue_size=queue_size,
            piece_source_factory=self._piece_source_factory(seed),
        )
        self.layout = self.core.observation_layout
        self.action_space = spaces.Discrete(len(API_ACTIONS))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.layout.size,), dtype=np.float32)
        self._seed = seed
        self._episode_index = 0
        self._shared_observation: SharedObservationBuffer | None = None
        if use_shared_memory:
            self._shared_observation = create_shared_observation_buffer(self.layout)

    def _piece_source_factory(self, seed: int | None):
        if self.piece_source == "classic_uniform":
            return lambda: classic_uniform_source(seed=seed)
        if self.piece_source == "seven_bag":
            return None
        raise ValueError(f"Unsupported piece source: {self.piece_source}")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = seed
            self._episode_index = 0
        effective_seed = self._seed if self._seed is None else self._seed + self._episode_index
        self._episode_index += 1
        self.core.piece_source_factory = self._piece_source_factory(effective_seed)
        observation, info = self.core.reset(seed=effective_seed, options=options)
        if self._shared_observation is not None:
            observation = snapshot_to_observation(
                info["snapshot"],
                out=self._shared_observation.array,
                layout=self.layout,
            )
        return observation, self._info_from_snapshot(info["snapshot"])

    def step(self, action):
        if isinstance(action, (int, np.integer)):
            action = API_ACTIONS[int(action)]
        observation, reward, terminated, truncated, info = self.core.step(action)
        snapshot = info["snapshot"]
        if self._shared_observation is not None:
            observation = snapshot_to_observation(snapshot, out=self._shared_observation.array, layout=self.layout)
        return observation, reward, terminated, truncated, self._info_from_snapshot(snapshot, info)

    def snapshot(self) -> dict:
        return self.core.snapshot()

    def close(self):
        if self._shared_observation is not None:
            self._shared_observation.close()
            try:
                self._shared_observation.unlink()
            except FileNotFoundError:
                pass
            self._shared_observation = None
        self.core.close()

    def render(self):  # pragma: no cover - optional interactive hook
        return self.core.game_state

    def _info_from_snapshot(self, snapshot: dict, step_info: dict | None = None) -> dict:
        return {
            "score": int(snapshot.get("score", 0)),
            "lines_cleared": int(snapshot.get("lines_cleared", 0)),
            "level": int(snapshot.get("level", 1)),
            "game_over": bool(snapshot.get("game_over", False)),
            "hold_used": bool(snapshot.get("hold_used", False)),
            "paused": bool(snapshot.get("paused", False)),
            "state": snapshot.get("app_state", "playing"),
            "reward_breakdown": step_info.get("reward_breakdown") if step_info else None,
        }


def make_vec_env(
    *,
    num_envs: int,
    difficulty,
    seed: int | None = None,
    queue_size: int = 5,
    use_shared_memory: bool = False,
    piece_source: str = "classic_uniform",
) -> DummyVecEnv | SubprocVecEnv:
    if num_envs <= 1:
        return DummyVecEnv(
            [
                lambda: TetrisGymEnv(
                    difficulty=difficulty,
                    seed=seed,
                    queue_size=queue_size,
                    use_shared_memory=use_shared_memory,
                    piece_source=piece_source,
                )
            ]
        )

    def make_env(rank: int) -> Callable[[], TetrisGymEnv]:
        def _init() -> TetrisGymEnv:
            env_seed = None if seed is None else seed + rank
            return TetrisGymEnv(
                difficulty=difficulty,
                seed=env_seed,
                queue_size=queue_size,
                use_shared_memory=use_shared_memory,
                piece_source=piece_source,
            )

        return _init

    return SubprocVecEnv([make_env(rank) for rank in range(num_envs)], start_method="spawn")
