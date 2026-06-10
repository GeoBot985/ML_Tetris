import copy

import numpy as np
import pytest

from ai_agent.environment import ACTION_NAMES, TetrisEnvironment, snapshot_to_observation
from ai_agent.policy import API_ACTIONS
from tetris.difficulty import NORMAL
from tetris.piece_source import classic_uniform_source


def source():
    while True:
        yield "O"


def test_snapshot_to_observation_is_fixed_size_and_normalized():
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)
    snapshot = env.snapshot()

    observation = snapshot_to_observation(snapshot)

    assert isinstance(observation, np.ndarray)
    assert observation.dtype == np.float32
    assert observation.ndim == 1
    assert observation.size > 0
    assert observation.min() >= 0.0
    assert observation.max() <= 1.0


def test_snapshot_to_observation_includes_queue_and_hold_features():
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)
    snapshot = copy.deepcopy(env.snapshot())
    baseline = snapshot_to_observation(snapshot)

    snapshot["next_queue"] = ["I", "T", "L", "S", "Z"]
    queue_variant = snapshot_to_observation(snapshot)
    snapshot["hold_piece"] = "O"
    snapshot["hold_used"] = True
    hold_variant = snapshot_to_observation(snapshot)

    assert queue_variant.shape == baseline.shape
    assert hold_variant.shape == baseline.shape
    assert not np.array_equal(baseline, queue_variant)
    assert not np.array_equal(queue_variant, hold_variant)


def test_environment_reset_returns_observation_and_snapshot():
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)

    observation, info = env.reset()

    assert observation.shape == snapshot_to_observation(info["snapshot"]).shape
    assert info["snapshot"]["app_state"] == "playing"
    assert env.game_state.active_piece.name == "O"


def test_environment_accepts_classic_uniform_piece_source_factory():
    env = TetrisEnvironment(piece_source_factory=lambda: classic_uniform_source(seed=123), difficulty=NORMAL)

    snapshot = env.snapshot()
    pieces = [snapshot["active_piece"]["name"], *snapshot["next_queue"]]

    assert len(pieces) == 6
    assert all(piece in {"I", "O", "T", "S", "Z", "J", "L"} for piece in pieces)


def test_environment_step_accepts_string_and_index_actions():
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)

    starting_x = env.game_state.active_x
    observation, reward, terminated, truncated, info = env.step("left")

    assert env.game_state.active_x == starting_x - 1
    assert observation.shape[0] > 0
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["command"] == "left"

    env.reset()
    env.game_state.active_x = starting_x
    # Integer actions are decoded via API_ACTIONS (the policy's action table),
    # not ACTION_NAMES — see TetrisEnvironment._command_for_action.
    observation, reward, terminated, truncated, info = env.step(API_ACTIONS.index("right"))

    assert env.game_state.active_x == starting_x + 1
    assert observation.shape[0] > 0
    assert info["command"] == "right"


def test_environment_step_noop_advances_gravity():
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)
    starting_y = env.game_state.active_y

    env.step("noop")

    assert env.game_state.active_y == starting_y + 1


def test_environment_rejects_unknown_actions():
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)

    with pytest.raises(ValueError):
        env.step("teleport")
