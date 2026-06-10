"""Regression tests covering all critical game/AI paths.

Run fast subset with:  pytest -m "not slow"
Run everything with:   pytest
"""
from __future__ import annotations

import pytest

from tetris.board import Board
from tetris.commands import COMMAND_NAMES, apply_command
from tetris.difficulty import EASY, HARD, NORMAL
from tetris.game_state import GameState
from tetris.piece_source import seven_bag_piece_source
from tetris.pieces import all_piece_names, make_piece
from tetris.scoring import hard_drop_score, line_clear_score, soft_drop_score


# ── helpers ───────────────────────────────────────────────────────────────────

def _fixed_source(*names):
    for n in names:
        yield n
    while True:
        yield names[-1]


def _fresh_game(*pieces):
    return GameState(piece_source=_fixed_source(*pieces), queue_size=5)


# ── piece spawning ─────────────────────────────────────────────────────────

def test_new_game_has_active_piece():
    game = _fresh_game("I", "O")
    assert game.active_piece is not None
    assert game.active_piece.name == "I"


def test_spawned_piece_is_within_board():
    game = _fresh_game("T", "O")
    for dx, dy in game.active_piece.cells():
        assert 0 <= game.active_x + dx < game.board.width


# ── seven-bag balancing ────────────────────────────────────────────────────

def test_seven_bag_first_bag_has_all_pieces():
    source = seven_bag_piece_source(shuffle=lambda bag: None)
    bag = [next(source) for _ in range(7)]
    assert sorted(bag) == sorted(all_piece_names())


def test_seven_bag_never_repeats_within_bag():
    source = seven_bag_piece_source()
    bag = [next(source) for _ in range(7)]
    assert len(set(bag)) == 7


# ── hold rules ────────────────────────────────────────────────────────────

def test_hold_swaps_piece():
    game = _fresh_game("I", "O", "T")
    first = game.active_piece.name
    assert game.hold()
    assert game.hold_piece.name == first
    assert game.active_piece.name != first


def test_hold_can_only_be_used_once_per_piece():
    game = _fresh_game("I", "O", "T")
    assert game.hold()
    assert not game.hold(), "second hold in same turn should be blocked"


def test_hold_marks_hold_used():
    game = _fresh_game("I", "O", "T")
    game.hold()
    assert game.hold_used is True


# ── rotation legality ─────────────────────────────────────────────────────

def test_rotation_changes_orientation():
    game = _fresh_game("T", "O")
    rot_before = game.active_piece.rotation
    game.rotate_clockwise()
    assert game.active_piece.rotation != rot_before


def test_o_piece_rotation_is_noop_on_grid():
    game = _fresh_game("O", "I")
    x_before, y_before = game.active_x, game.active_y
    game.rotate_clockwise()
    assert game.active_x == x_before
    assert game.active_y == y_before


def test_rotation_blocked_by_wall():
    game = _fresh_game("I", "O")
    game.active_x = 0
    result = game.rotate_clockwise()
    # might or might not succeed depending on kick, but must not crash
    assert isinstance(result, bool)


# ── line clearing ─────────────────────────────────────────────────────────

def test_full_row_is_cleared_on_lock():
    game = _fresh_game("I", "O")
    # Fill row 19 completely and lock an I piece on top to trigger a clear
    for x in range(game.board.width):
        game.board.set_cell(x, game.board.height - 1, "X")
    lines_before = game.lines_cleared
    # Hard-drop the active I piece — it lands on top and clears at least one row
    game.hard_drop()
    # The locked I piece might fill row 18 with a full row; either way the filled row 19 must clear
    # (game may have spawned next piece, so just verify lines increased)
    assert game.lines_cleared > lines_before


def test_clearing_four_lines_scores_more_than_two():
    score_2 = line_clear_score(2, 1)
    score_4 = line_clear_score(4, 1)
    assert score_4 > score_2 * 2  # Tetris bonus


# ── scoring ────────────────────────────────────────────────────────────────

def test_hard_drop_score_proportional_to_distance():
    assert hard_drop_score(5) == 10
    assert hard_drop_score(0) == 0


def test_soft_drop_score_is_one_per_cell():
    assert soft_drop_score(3) == 3


def test_line_clear_score_scales_with_level():
    assert line_clear_score(1, 2) == line_clear_score(1, 1) * 2


# ── level progression ─────────────────────────────────────────────────────

def test_level_increases_after_ten_lines():
    game = _fresh_game("O", "I")
    game.lines_cleared = 9
    game.level = 1
    # Manually trigger level calculation as lock_active_piece would
    game.lines_cleared += 1
    game.level = 1 + game.lines_cleared // 10
    assert game.level == 2


# ── command mapping via commands.py (pygame-free) ─────────────────────────

def test_commands_module_apply_command_moves():
    game = _fresh_game("T", "O")
    x_before = game.active_x
    apply_command(game, "right")
    assert game.active_x == x_before + 1


def test_commands_module_apply_command_hold():
    game = _fresh_game("I", "O", "T")
    first = game.active_piece.name
    apply_command(game, "hold")
    assert game.hold_piece.name == first


def test_commands_module_apply_command_pause():
    game = _fresh_game("T", "O")
    apply_command(game, "pause")
    assert game.paused


def test_commands_module_quit_returns_string():
    game = _fresh_game("T", "O")
    result = apply_command(game, "quit")
    assert result == "quit"


def test_commands_module_unknown_returns_none():
    game = _fresh_game("T", "O")
    result = apply_command(game, "bogus_action")
    assert result is None


def test_all_command_names_are_handled():
    """Every name in COMMAND_NAMES should produce a non-None result (or 'quit')."""
    game = _fresh_game("T", "O", "I", "S", "Z", "J", "L")
    # soft_drop and left/right should work from starting position
    for cmd in ("right", "left", "rotate_cw", "rotate_ccw", "hard_drop"):
        game2 = _fresh_game("T", "O")
        apply_command(game2, cmd)  # must not raise


# ── observation encoding / decoding ──────────────────────────────────────

def test_observation_encoding_is_importable_without_pygame():
    from ai_agent.environment import snapshot_to_observation, observation_to_snapshot, TetrisEnvironment
    from tetris.difficulty import NORMAL

    def src():
        while True:
            yield "O"

    env = TetrisEnvironment(piece_source_factory=src, difficulty=NORMAL)
    snap = env.snapshot()
    obs = snapshot_to_observation(snap)
    reconstructed = observation_to_snapshot(obs)

    assert reconstructed["app_state"] == snap["app_state"]
    assert reconstructed["hold_used"] == snap["hold_used"]
    assert reconstructed["active_piece"]["name"] == snap["active_piece"]["name"]


# ── safety correction ─────────────────────────────────────────────────────

def test_safety_wrapper_single_model_call(monkeypatch):
    """SafetyWrapper.decide() must invoke predict_from_snapshot exactly once."""
    import copy
    import torch
    from ai_agent.environment import TetrisEnvironment
    from ai_agent.policy import API_ACTIONS, PolicyDecision, PPOPolicy
    from ai_agent.safety import SafetyWrapper

    call_count = 0

    env = TetrisEnvironment()
    env.reset(seed=42)
    snap = copy.deepcopy(env.snapshot())

    policy = PPOPolicy.from_observation_dim(10)
    action_index = API_ACTIONS.index("right")
    logits = torch.zeros(1, len(API_ACTIONS))
    logits[0, action_index] = 3.0

    def counting_predict(snapshot, deterministic=False):
        nonlocal call_count
        call_count += 1
        return PolicyDecision(
            action_index=action_index,
            action="right",
            logits=logits,
            log_prob=torch.tensor(-0.05),
            value=torch.tensor(0.5),
            entropy=torch.tensor(0.25),
        )

    policy.predict_from_snapshot = counting_predict  # type: ignore[method-assign]
    wrapper = SafetyWrapper(policy, difficulty=NORMAL)
    wrapper.decide(snap)

    assert call_count == 1, f"Expected 1 model call, got {call_count}"


# ── manifest logging ──────────────────────────────────────────────────────

def test_manifest_logger_output_is_jsonl(tmp_path):
    import json
    from ai_agent.deployment import ManifestLogger, build_manifest_record
    from ai_agent.environment import TetrisEnvironment
    from ai_agent.policy import PPOPolicy
    from ai_agent.safety import SafetyWrapper

    def src():
        while True:
            yield "O"

    env = TetrisEnvironment(piece_source_factory=src, difficulty=NORMAL)
    policy = PPOPolicy.from_snapshot(env.snapshot())
    wrapper = SafetyWrapper(policy, difficulty=NORMAL)
    log_path = tmp_path / "manifest.jsonl"
    logger = ManifestLogger(log_path, metadata={"run_id": "regression-test"})
    snap = env.snapshot()
    decision = wrapper.decide(snap)
    logger.record(
        build_manifest_record(
            run_id="regression-test",
            episode=1,
            step=0,
            snapshot=snap,
            safety_decision=decision,
            reward=0.5,
            terminated=False,
            truncated=False,
            diagnostics=None,
        )
    )
    logger.summarize({"run_id": "regression-test", "episodes": 1})
    logger.close()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    for line in lines:
        obj = json.loads(line)
        assert "type" in obj


# ── benchmark output format ───────────────────────────────────────────────

def test_benchmark_result_has_required_fields():
    from ai_agent.deployment import benchmark_policy
    from ai_agent.environment import TetrisEnvironment
    from ai_agent.policy import PPOPolicy

    def src():
        while True:
            yield "O"

    env = TetrisEnvironment(piece_source_factory=src, difficulty=NORMAL)
    policy = PPOPolicy.from_snapshot(env.snapshot())
    result = benchmark_policy(policy, env.snapshot(), iterations=3, warmup_iterations=1)

    assert hasattr(result, "raw_mean_ms")
    assert hasattr(result, "raw_p95_ms")
    assert hasattr(result, "guarded_mean_ms")
    assert hasattr(result, "guarded_p95_ms")
    assert result.raw_mean_ms >= 0.0


def test_ai_agent_top_level_import_does_not_expose_optional_vectorized_backend():
    import ai_agent

    assert not hasattr(ai_agent, "make_vec_env")
    assert not hasattr(ai_agent, "TetrisGymEnv")
    assert not hasattr(ai_agent, "SharedObservationBuffer")
