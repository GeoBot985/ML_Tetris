"""Tests for the placement-level RL stack (env, CNN policy, coach expert, trainer)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ai_agent.placement import (
    AUX_DIM,
    BOARD_H,
    BOARD_W,
    N_SLOTS,
    CNNActorCritic,
    PlacementEnv,
    PlacementPolicy,
    PlacementTrainConfig,
    aux_features,
    board_planes,
    coach_slot,
    legal_mask,
    legal_placements,
    load_placement_policy,
    train_placement_policy,
)
from tetris.difficulty import NORMAL


def test_action_space_is_rotations_times_columns():
    assert N_SLOTS == 4 * BOARD_W


def test_observation_shapes():
    env = PlacementEnv(difficulty=NORMAL)
    env.reset(seed=1)
    planes = board_planes(env.game_state)
    aux = aux_features(env.game_state)
    assert planes.shape == (1, BOARD_H, BOARD_W)
    assert aux.shape == (AUX_DIM,)


def test_legal_mask_matches_placements():
    env = PlacementEnv(difficulty=NORMAL)
    env.reset(seed=2)
    mask = legal_mask(env.game_state)
    placements = legal_placements(env.game_state)
    assert mask.sum() == len(placements)
    assert mask.any()
    for slot in placements:
        assert mask[slot]


def test_coach_slot_is_legal():
    env = PlacementEnv(difficulty=NORMAL)
    env.reset(seed=3)
    slot = coach_slot(env.game_state, NORMAL)
    assert slot is not None
    assert legal_mask(env.game_state)[slot]


def test_coach_plays_meaningfully_longer_than_random():
    """Sanity: the coach must survive much longer than random placement."""
    def play(use_coach: bool, seed: int) -> int:
        env = PlacementEnv(difficulty=NORMAL, max_pieces=200)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        while True:
            if use_coach:
                slot = env.coach_slot()
            else:
                legal = np.flatnonzero(env.legal_mask())
                slot = int(rng.choice(legal)) if len(legal) else None
            if slot is None:
                break
            if env.step(slot).terminated:
                break
        return env.game_state.lines_cleared

    coach_lines = np.mean([play(True, s) for s in range(3)])
    random_lines = np.mean([play(False, s) for s in range(3)])
    assert coach_lines > random_lines + 1.0


def test_step_terminates_on_game_over():
    env = PlacementEnv(difficulty=NORMAL, max_pieces=500)
    env.reset(seed=5)
    terminated = False
    for _ in range(500):
        legal = np.flatnonzero(env.legal_mask())
        if len(legal) == 0:
            break
        # Always stack in column 0 rotation 0 to force a quick top-out
        slot = int(legal[0])
        if env.step(slot).terminated:
            terminated = True
            break
    assert terminated


def test_cnn_forward_and_masked_action():
    model = CNNActorCritic()
    env = PlacementEnv(difficulty=NORMAL)
    env.reset(seed=6)
    planes = torch.as_tensor(board_planes(env.game_state))
    aux = torch.as_tensor(aux_features(env.game_state))
    logits, value = model(planes, aux)
    assert logits.shape == (1, N_SLOTS)
    assert value.shape == (1,)
    policy = PlacementPolicy(model)
    slot = policy.act(env.game_state, deterministic=True)
    assert slot is not None
    assert legal_mask(env.game_state)[slot]


def test_checkpoint_round_trip(tmp_path):
    model = CNNActorCritic()
    from ai_agent.placement import save_placement_checkpoint, PlacementSummary

    path = tmp_path / "p.pt"
    save_placement_checkpoint(path, model, PlacementTrainConfig(), PlacementSummary(best_eval_lines=3.0))
    policy = load_placement_policy(path)
    env = PlacementEnv(difficulty=NORMAL)
    env.reset(seed=7)
    assert policy.act(env.game_state) is not None


@pytest.mark.slow
def test_training_smoke_learns_and_saves(tmp_path):
    """Tiny BC(+DAgger+PPO) run must produce a checkpoint and record an eval curve."""
    config = PlacementTrainConfig(
        bc_states=400,
        bc_epochs=2,
        dagger_iterations=1,
        dagger_episodes=3,
        ppo_updates=2,
        ppo_rollout_episodes=2,
        eval_episodes=2,
        max_pieces=80,
        checkpoint_path=tmp_path / "placement.pt",
        seed=0,
    )
    summary = train_placement_policy(config)
    assert config.checkpoint_path.exists()
    assert summary.checkpoint_path == config.checkpoint_path
    # one BC eval + one per DAgger iter (+ PPO evals)
    assert len(summary.eval_lines) >= 1 + config.dagger_iterations
