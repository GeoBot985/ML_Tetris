from __future__ import annotations

import copy

import torch

from ai_agent.environment import TetrisEnvironment
from ai_agent.policy import API_ACTIONS, PolicyDecision, PPOPolicy
from ai_agent.safety import SafetyWrapper


def build_stub_policy(action: str, value: float = 0.0, entropy: float = 0.25) -> PPOPolicy:
    policy = PPOPolicy.from_observation_dim(10)
    action_index = API_ACTIONS.index(action)
    logits = torch.full((1, len(API_ACTIONS)), -2.0)
    logits[0, action_index] = 3.0

    def predict_from_snapshot(snapshot, deterministic: bool = False):
        return PolicyDecision(
            action_index=action_index,
            action=action,
            logits=logits,
            log_prob=torch.tensor(-0.05),
            value=torch.tensor(value),
            entropy=torch.tensor(entropy),
        )

    policy.predict_from_snapshot = predict_from_snapshot  # type: ignore[method-assign]
    return policy


def test_safety_wrapper_corrects_illegal_action():
    env = TetrisEnvironment()
    env.reset(seed=1)
    snapshot = copy.deepcopy(env.snapshot())
    snapshot["app_state"] = "playing"
    snapshot["active_piece"]["x"] = 0

    wrapper = SafetyWrapper(build_stub_policy("left"), difficulty=env.difficulty)

    decision = wrapper.decide(snapshot)

    assert decision.corrected is True
    assert decision.correction_reason == "illegal_action"
    assert decision.model_decision.action == "left"
    assert wrapper.is_legal_action(snapshot, decision.executed_action) is True


def test_safety_wrapper_corrects_high_risk_state(monkeypatch):
    env = TetrisEnvironment()
    env.reset(seed=2)
    snapshot = copy.deepcopy(env.snapshot())
    snapshot["app_state"] = "playing"

    wrapper = SafetyWrapper(build_stub_policy("right"), difficulty=env.difficulty)
    monkeypatch.setattr(wrapper, "risk_score", lambda *_args, **_kwargs: 0.99)

    decision = wrapper.decide(snapshot)

    assert decision.corrected is True
    assert decision.correction_reason == "high_risk"
    assert decision.model_decision.action == "right"
    assert wrapper.is_legal_action(snapshot, decision.executed_action) is True
