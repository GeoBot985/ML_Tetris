import pytest
import torch

from ai_agent.human_hints import count_human_hints, load_human_hints, write_human_hint
from ai_agent.policy import API_ACTIONS
from ai_agent.training import TrainingConfig, coach_action, compute_gae, load_policy_from_checkpoint, train_policy
from tetris.difficulty import NORMAL


def source():
    while True:
        yield "O"


def test_coach_action_returns_valid_api_command():
    from ai_agent.environment import TetrisEnvironment

    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)
    action, score, breakdown = coach_action(env.snapshot(), difficulty=NORMAL)

    assert action in API_ACTIONS
    assert isinstance(score, float)
    assert breakdown.total == score or isinstance(breakdown.total, float)


def test_compute_gae_returns_advantages_and_returns():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.tensor([0.0, 0.0, 1.0])
    last_value = torch.tensor(0.0)

    advantages, returns = compute_gae(rewards, values, dones, last_value, gamma=1.0, lambda_=1.0)

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert torch.allclose(returns, advantages + values)


def test_human_hints_round_trip(tmp_path):
    from ai_agent.environment import TetrisEnvironment

    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)
    path = tmp_path / "human_hints.jsonl"

    assert write_human_hint(path, env.snapshot(), "left", difficulty=NORMAL.name)
    assert not write_human_hint(path, env.snapshot(), "pause", difficulty=NORMAL.name)

    hints = load_human_hints(path)
    assert hints is not None
    observations, actions = hints
    assert count_human_hints(path) == 1
    assert observations.shape[0] == 1
    assert actions.tolist() == [API_ACTIONS.index("left")]


@pytest.mark.slow
def test_training_smoke_creates_checkpoint_and_reloadable_policy(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    log_path = tmp_path / "metrics.jsonl"
    feedback_path = tmp_path / "feedback.md"
    config = TrainingConfig(
        episodes=1,
        max_steps=4,
        rollout_steps=4,
        ppo_epochs=1,
        batch_size=4,
        checkpoint_path=checkpoint,
        log_path=log_path,
        feedback_path=feedback_path,
        evaluation_interval=1,
        evaluation_episodes=1,
        seed=1,
    )

    summary = train_policy(config)
    policy = load_policy_from_checkpoint(checkpoint)

    assert checkpoint.exists()
    assert log_path.exists()
    assert feedback_path.exists()
    assert summary.checkpoint_path == checkpoint
    assert len(summary.episodes) == 1
    from ai_agent.environment import TetrisEnvironment

    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)
    snapshot = env.snapshot()
    assert policy.act_from_snapshot(snapshot).action in API_ACTIONS
