from __future__ import annotations

import pytest

from ai_agent.environment import TetrisEnvironment, observation_to_snapshot, snapshot_to_observation
from ai_agent.training import TrainingConfig, load_policy_from_checkpoint, train_policy
from ai_agent.vectorized import make_vec_env
from tetris.difficulty import NORMAL


def source():
    while True:
        yield "O"


def test_observation_round_trip_preserves_key_state():
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)
    snapshot = env.snapshot()

    observation = snapshot_to_observation(snapshot)
    reconstructed = observation_to_snapshot(observation)

    assert reconstructed["hold_used"] == snapshot["hold_used"]
    assert reconstructed["next_queue"] == snapshot["next_queue"]
    assert reconstructed["hold_piece"] == snapshot["hold_piece"]
    assert reconstructed["active_piece"]["name"] == snapshot["active_piece"]["name"]
    assert reconstructed["active_piece"]["rotation"] == snapshot["active_piece"]["rotation"]


@pytest.mark.slow
def test_make_vec_env_supports_subprocess_vectorization():
    vec_env = make_vec_env(num_envs=2, difficulty=NORMAL, seed=1)
    try:
        observations = vec_env.reset()
        next_observations, rewards, dones, infos = vec_env.step([0, 0])

        assert observations.shape[0] == 2
        assert next_observations.shape[0] == 2
        assert len(rewards) == 2
        assert len(dones) == 2
        assert len(infos) == 2
    finally:
        vec_env.close()


@pytest.mark.slow
def test_parallel_training_smoke_creates_checkpoint(tmp_path):
    checkpoint = tmp_path / "parallel_policy.pt"
    config = TrainingConfig(
        episodes=1,
        max_steps=4,
        checkpoint_path=checkpoint,
        log_path=tmp_path / "parallel_metrics.jsonl",
        feedback_path=tmp_path / "parallel_feedback.md",
        evaluation_interval=1,
        evaluation_episodes=1,
        seed=1,
        parallel_envs=2,
        use_shared_memory=False,
        verify_quantized_checkpoint=False,
    )

    summary = train_policy(config)
    policy = load_policy_from_checkpoint(checkpoint, quantize=True)

    assert checkpoint.exists()
    assert len(summary.episodes) == 1
    assert policy is not None
