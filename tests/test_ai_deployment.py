from __future__ import annotations

import json

import pytest

from ai_agent.deployment import ManifestLogger, benchmark_policy, build_manifest_record
from ai_agent.diagnostics import build_decision_diagnostics
from ai_agent.environment import TetrisEnvironment
from ai_agent.policy import API_ACTIONS, PolicyDecision, PPOPolicy
from ai_agent.safety import SafetyWrapper
from ai_agent.training import load_policy_from_checkpoint, train_policy, TrainingConfig
from tetris.difficulty import NORMAL


def source():
    while True:
        yield "O"


@pytest.mark.slow
def test_quantized_policy_loads_and_predicts_valid_action(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    config = TrainingConfig(
        episodes=1,
        max_steps=4,
        checkpoint_path=checkpoint,
        log_path=tmp_path / "metrics.jsonl",
        feedback_path=tmp_path / "feedback.md",
        evaluation_interval=1,
        evaluation_episodes=1,
        seed=1,
    )

    train_policy(config)
    policy = load_policy_from_checkpoint(checkpoint, quantize=True)
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)

    decision = policy.act_from_snapshot(env.snapshot(), deterministic=True)

    assert checkpoint.exists()
    assert decision.action in API_ACTIONS


def test_manifest_logger_records_decision_stream(tmp_path):
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)
    policy = PPOPolicy.from_snapshot(env.snapshot())
    wrapper = SafetyWrapper(policy, difficulty=NORMAL)
    manifest_path = tmp_path / "manifest.jsonl"
    logger = ManifestLogger(manifest_path, metadata={"run_id": "test-run", "mode": "headless"})

    snapshot = env.snapshot()
    decision = wrapper.decide(snapshot)
    diagnostics = build_decision_diagnostics(
        executed_action=decision.executed_action,
        model_decision=decision.model_decision,
        fallback_used=decision.corrected,
        correction_reason=decision.correction_reason,
        risk_score=decision.risk_score,
    )
    logger.record(
        build_manifest_record(
            run_id=logger.run_id or "test-run",
            episode=1,
            step=0,
            snapshot=snapshot,
            safety_decision=decision,
            reward=1.5,
            terminated=False,
            truncated=False,
            diagnostics=diagnostics,
            latency_ms=0.12,
        )
    )
    logger.record_event("episode_summary", {"run_id": logger.run_id or "test-run", "episode": 1, "steps": 1})
    logger.summarize({"run_id": logger.run_id or "test-run", "episodes": 1})
    logger.close()

    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    start_event = json.loads(lines[0])
    decision_event = json.loads(lines[1])
    summary_event = json.loads(lines[-1])

    assert start_event["type"] == "run_start"
    assert decision_event["type"] == "decision"
    assert decision_event["executed_action"] in API_ACTIONS
    assert decision_event["state_hash"]
    assert summary_event["type"] == "run_summary"


def test_benchmark_policy_reports_latency():
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)
    policy = PPOPolicy.from_snapshot(env.snapshot())

    result = benchmark_policy(policy, env.snapshot(), iterations=5, warmup_iterations=1)

    assert result.raw_mean_ms >= 0.0
    assert result.guarded_mean_ms >= 0.0
    assert result.raw_p95_ms >= 0.0
    assert result.guarded_p95_ms >= 0.0
