from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random
from typing import Any, Callable, Iterable

import numpy as np

from tetris.api import API_COMMANDS, build_snapshot
from tetris.commands import apply_command
from tetris.difficulty import EASY, HARD, NORMAL, Difficulty
from tetris.game_state import GameState
from tetris.piece_source import seven_bag_piece_source
from tetris.pieces import make_piece

from .rewards import calculate_reward


ACTION_NAMES = (
    "noop",
    "left",
    "right",
    "soft_drop",
    "hard_drop",
    "rotate_cw",
    "rotate_ccw",
    "pause",
    "restart",
    "hold",
)

PIECE_NAMES = ("I", "O", "T", "S", "Z", "J", "L")
APP_STATES = ("start", "playing", "game_over")
PIECE_VALUE_MAP = {name: (index + 1) / len(PIECE_NAMES) for index, name in enumerate(PIECE_NAMES)}
VALUE_TO_PIECE = {value: name for name, value in PIECE_VALUE_MAP.items()}


@dataclass(frozen=True)
class ObservationLayout:
    board_width: int
    board_height: int
    queue_size: int
    board_slice: slice
    active_mask_slice: slice
    active_piece_slice: slice
    active_rotation_index: int
    hold_piece_slice: slice
    hold_used_index: int
    queue_slice: slice
    score_index: int
    level_index: int
    lines_index: int
    gravity_index: int
    paused_index: int
    game_over_index: int
    app_state_slice: slice
    size: int


@lru_cache(maxsize=8)
def build_observation_layout(board_width: int = 10, board_height: int = 20, queue_size: int = 5) -> ObservationLayout:
    cursor = 0
    board_size = board_width * board_height
    board_slice = slice(cursor, cursor + board_size)
    cursor += board_size
    active_mask_slice = slice(cursor, cursor + board_size)
    cursor += board_size
    active_piece_slice = slice(cursor, cursor + len(PIECE_NAMES))
    cursor += len(PIECE_NAMES)
    active_rotation_index = cursor
    cursor += 1
    hold_piece_slice = slice(cursor, cursor + len(PIECE_NAMES))
    cursor += len(PIECE_NAMES)
    hold_used_index = cursor
    cursor += 1
    queue_slice = slice(cursor, cursor + queue_size * len(PIECE_NAMES))
    cursor += queue_size * len(PIECE_NAMES)
    score_index = cursor
    level_index = cursor + 1
    lines_index = cursor + 2
    gravity_index = cursor + 3
    cursor += 4
    paused_index = cursor
    game_over_index = cursor + 1
    cursor += 2
    app_state_slice = slice(cursor, cursor + len(APP_STATES))
    cursor += len(APP_STATES)
    return ObservationLayout(
        board_width=board_width,
        board_height=board_height,
        queue_size=queue_size,
        board_slice=board_slice,
        active_mask_slice=active_mask_slice,
        active_piece_slice=active_piece_slice,
        active_rotation_index=active_rotation_index,
        hold_piece_slice=hold_piece_slice,
        hold_used_index=hold_used_index,
        queue_slice=queue_slice,
        score_index=score_index,
        level_index=level_index,
        lines_index=lines_index,
        gravity_index=gravity_index,
        paused_index=paused_index,
        game_over_index=game_over_index,
        app_state_slice=app_state_slice,
        size=cursor,
    )


def _one_hot(value: str | None, options: Iterable[str]) -> list[float]:
    options = tuple(options)
    vector = [0.0] * len(options)
    if value in options:
        vector[options.index(value)] = 1.0
    return vector


def _piece_value(cell: str | None) -> float:
    if cell is None:
        return 0.0
    if cell not in PIECE_NAMES:
        return 1.0
    return (PIECE_NAMES.index(cell) + 1) / len(PIECE_NAMES)


def _normalize(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(np.clip(value / scale, 0.0, 1.0))


def _extract_active_mask(snapshot: dict[str, Any], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    active_piece = snapshot.get("active_piece")
    if not active_piece:
        return mask
    origin_x = int(active_piece.get("x", 0))
    origin_y = int(active_piece.get("y", 0))
    for dx, dy in active_piece.get("cells", []):
        x = origin_x + int(dx)
        y = origin_y + int(dy)
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = 1.0
    return mask


def _decode_piece_value(value: float) -> str | None:
    if value <= 0.0:
        return None
    if value >= 0.99:
        return None
    index = int(round(value * len(PIECE_NAMES))) - 1
    if 0 <= index < len(PIECE_NAMES):
        return PIECE_NAMES[index]
    return None


def _write_one_hot(buffer: np.ndarray, start: int, options: tuple[str, ...], value: str | None) -> None:
    end = start + len(options)
    buffer[start:end] = 0.0
    if value in options:
        buffer[start + options.index(value)] = 1.0


def _decode_one_hot(values: np.ndarray, options: tuple[str, ...]) -> str | None:
    if values.size == 0:
        return None
    index = int(np.argmax(values))
    if values[index] <= 0.0:
        return None
    if 0 <= index < len(options):
        return options[index]
    return None


def _infer_piece_origin(mask: np.ndarray, piece_name: str | None, rotation: int) -> tuple[int, int]:
    if piece_name is None:
        return 0, 0
    cells = make_piece(piece_name, rotation).cells()
    positions = np.argwhere(mask > 0.5)
    if positions.size == 0:
        return 0, 0
    height, width = mask.shape
    for y, x in positions:
        for dx, dy in cells:
            origin_x = int(x) - int(dx)
            origin_y = int(y) - int(dy)
            if origin_x < 0 or origin_y < 0:
                continue
            fits = True
            for cell_dx, cell_dy in cells:
                px = origin_x + int(cell_dx)
                py = origin_y + int(cell_dy)
                if not (0 <= px < width and 0 <= py < height and mask[py, px] > 0.5):
                    fits = False
                    break
            if fits:
                return origin_x, origin_y
    min_x = int(np.min(positions[:, 1]))
    min_y = int(np.min(positions[:, 0]))
    min_dx = min(dx for dx, _ in cells)
    min_dy = min(dy for _, dy in cells)
    return max(0, min_x - min_dx), max(0, min_y - min_dy)


def snapshot_to_observation(
    snapshot: dict[str, Any],
    board_width: int = 10,
    board_height: int = 20,
    queue_size: int = 5,
    *,
    out: np.ndarray | None = None,
    layout: ObservationLayout | None = None,
) -> np.ndarray:
    layout = layout or build_observation_layout(board_width, board_height, queue_size)
    if out is None:
        out = np.zeros(layout.size, dtype=np.float32)
    else:
        if out.shape != (layout.size,):
            raise ValueError(f"Observation buffer has shape {out.shape}, expected {(layout.size,)}")
        out.fill(0.0)

    locked_board = snapshot.get("locked_board") or snapshot.get("board") or []
    board_rows = min(layout.board_height, len(locked_board))
    for y in range(board_rows):
        row = locked_board[y]
        for x in range(min(layout.board_width, len(row))):
            out[layout.board_slice.start + y * layout.board_width + x] = _piece_value(row[x])

    active_mask = _extract_active_mask(snapshot, layout.board_width, layout.board_height)
    mask_offset = layout.active_mask_slice.start
    for y in range(layout.board_height):
        row_offset = mask_offset + y * layout.board_width
        out[row_offset : row_offset + layout.board_width] = active_mask[y, : layout.board_width]

    active_piece = snapshot.get("active_piece") or {}
    active_name = active_piece.get("name")
    active_rotation = int(active_piece.get("rotation", 0)) % 4
    _write_one_hot(out, layout.active_piece_slice.start, PIECE_NAMES, active_name)
    out[layout.active_rotation_index] = float(active_rotation) / 3.0

    _write_one_hot(out, layout.hold_piece_slice.start, PIECE_NAMES, snapshot.get("hold_piece"))
    out[layout.hold_used_index] = 1.0 if snapshot.get("hold_used") else 0.0

    queue = snapshot.get("next_queue") or []
    queue_offset = layout.queue_slice.start
    for index in range(layout.queue_size):
        piece_name = queue[index] if index < len(queue) else None
        _write_one_hot(out, queue_offset + index * len(PIECE_NAMES), PIECE_NAMES, piece_name)

    out[layout.score_index] = _normalize(float(snapshot.get("score", 0)), 100000.0)
    out[layout.level_index] = _normalize(float(snapshot.get("level", 1)), 30.0)
    out[layout.lines_index] = _normalize(float(snapshot.get("lines_cleared", 0)), 200.0)
    out[layout.gravity_index] = _normalize(float(snapshot.get("gravity_ms", 0)), 2000.0)
    out[layout.paused_index] = 1.0 if snapshot.get("paused") else 0.0
    out[layout.game_over_index] = 1.0 if snapshot.get("game_over") else 0.0
    _write_one_hot(out, layout.app_state_slice.start, APP_STATES, snapshot.get("app_state"))
    return out


def observation_to_snapshot(observation: np.ndarray, *, layout: ObservationLayout | None = None) -> dict[str, Any]:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    layout = layout or build_observation_layout()
    if obs.size != layout.size:
        raise ValueError(f"Observation has size {obs.size}, expected {layout.size}")

    board_values = obs[layout.board_slice].reshape(layout.board_height, layout.board_width)
    locked_board = [[_decode_piece_value(float(value)) for value in row] for row in board_values]

    active_mask = obs[layout.active_mask_slice].reshape(layout.board_height, layout.board_width)
    active_name = _decode_one_hot(obs[layout.active_piece_slice], PIECE_NAMES)
    active_rotation = int(round(float(obs[layout.active_rotation_index]) * 3.0)) % 4
    active_x, active_y = _infer_piece_origin(active_mask, active_name, active_rotation)
    active_piece = None
    if active_name is not None:
        active_piece = {
            "name": active_name,
            "rotation": active_rotation,
            "x": active_x,
            "y": active_y,
            "cells": list(make_piece(active_name, active_rotation).cells()),
        }

    queue = []
    for index in range(layout.queue_size):
        start = layout.queue_slice.start + index * len(PIECE_NAMES)
        piece_name = _decode_one_hot(obs[start : start + len(PIECE_NAMES)], PIECE_NAMES)
        if piece_name is not None:
            queue.append(piece_name)

    app_state = _decode_one_hot(obs[layout.app_state_slice], APP_STATES) or ("game_over" if obs[layout.game_over_index] > 0.5 else "playing")

    snapshot = {
        "app_state": app_state,
        "locked_board": locked_board,
        "board": [row[:] for row in locked_board],
        "active_piece": active_piece,
        "next_queue": queue,
        "hold_piece": _decode_one_hot(obs[layout.hold_piece_slice], PIECE_NAMES),
        "hold_used": bool(obs[layout.hold_used_index] > 0.5),
        "score": int(round(float(obs[layout.score_index]) * 100000)),
        "level": int(round(float(obs[layout.level_index]) * 30)),
        "lines_cleared": int(round(float(obs[layout.lines_index]) * 200)),
        "gravity_ms": int(round(float(obs[layout.gravity_index]) * 2000)),
        "paused": bool(obs[layout.paused_index] > 0.5),
        "game_over": bool(obs[layout.game_over_index] > 0.5),
    }
    return snapshot


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class TetrisEnvironment:
    action_names = ACTION_NAMES

    def __init__(
        self,
        difficulty: Difficulty = NORMAL,
        piece_source_factory: Callable[[], Iterable[str]] | None = None,
        queue_size: int = 5,
    ):
        self.difficulty = difficulty
        self.piece_source_factory = piece_source_factory
        self.queue_size = queue_size
        self.board_width = 10
        self.board_height = 20
        self.observation_layout = build_observation_layout(self.board_width, self.board_height, self.queue_size)
        self._observation_buffer = np.zeros(self.observation_layout.size, dtype=np.float32)
        self._seed: int | None = None
        self.game_state = self._new_game_state()
        self._last_snapshot = self.snapshot()

    def _new_game_state(self) -> GameState:
        if self.piece_source_factory is None:
            rng = random.Random(self._seed)
            return GameState(piece_source=seven_bag_piece_source(shuffle=rng.shuffle), queue_size=self.queue_size, difficulty=self.difficulty)
        return GameState(piece_source=self.piece_source_factory(), queue_size=self.queue_size, difficulty=self.difficulty)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._seed = seed
        self.game_state = self._new_game_state()
        snapshot = self.snapshot()
        self._last_snapshot = snapshot
        observation = snapshot_to_observation(snapshot, out=self._observation_buffer, layout=self.observation_layout)
        return observation.copy(), {"snapshot": snapshot}

    def _command_for_action(self, action: int | str) -> str:
        if isinstance(action, str):
            if action not in ACTION_NAMES:
                if action in {"start", "quit"}:
                    return "noop"
                raise ValueError(f"Unknown action: {action}")
            return action
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError(f"Action must be an int or str, got {type(action)!r}")
        # Integer actions index the SAME table the policy emits from (API_ACTIONS),
        # not ACTION_NAMES. The two tables differ in length and order; decoding an
        # int via ACTION_NAMES silently scrambles every action (see ai_agent.policy).
        from .policy import API_ACTIONS

        try:
            command = API_ACTIONS[int(action)]
        except IndexError as exc:
            raise ValueError(f"Unknown action index: {action}") from exc
        if command in {"start", "quit"}:
            return "noop"
        return command

    def snapshot(self) -> dict[str, Any]:
        app_state = "game_over" if self.game_state.game_over else ("playing" if not self.game_state.paused else "playing")
        selected_difficulty = self.difficulty
        difficulty_order = [EASY, NORMAL, HARD]
        return build_snapshot(app_state, self.game_state, selected_difficulty, difficulty_order)

    # Control commands are not legitimate RL actions: `restart` would wipe the
    # board mid-episode with no penalty or termination (a reward-hack escape
    # hatch), and `pause` freezes gravity so the episode never ends. The RL env
    # ignores them so the agent can only ever play the game.
    _RL_IGNORED_COMMANDS = frozenset({"pause", "restart", "start", "quit"})

    def step(self, action: int | str):
        previous_snapshot = self._last_snapshot
        command = self._command_for_action(action)
        if command in self._RL_IGNORED_COMMANDS:
            command = "noop"

        if command != "noop" and command in API_COMMANDS:
            apply_command(self.game_state, command)

        if command != "hard_drop" and not self.game_state.game_over:
            self.game_state.gravity_tick()

        snapshot = self.snapshot()
        self._last_snapshot = snapshot
        reward_breakdown = calculate_reward(previous_snapshot, snapshot)
        observation = snapshot_to_observation(snapshot, out=self._observation_buffer, layout=self.observation_layout)
        terminated = self.game_state.game_over
        truncated = False
        info = {"snapshot": snapshot, "command": command, "reward_breakdown": reward_breakdown}
        return observation.copy(), reward_breakdown.total, terminated, truncated, info

    def close(self):
        return None
