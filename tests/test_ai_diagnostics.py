import torch

from ai_agent.diagnostics import build_decision_diagnostics
from ai_agent.policy import API_ACTIONS, PolicyDecision


def test_build_decision_diagnostics_ranks_logits_and_tracks_fallback():
    logits = torch.tensor([[0.1, 0.9, 0.2, -0.1, 0.05, 0.04, 0.3, 0.0, -0.5, -0.2, -1.0]])
    decision = PolicyDecision(
        action_index=1,
        action="right",
        logits=logits,
        log_prob=torch.tensor(-0.1),
        value=torch.tensor(0.75),
        entropy=torch.tensor(1.2),
    )

    diagnostics = build_decision_diagnostics("hold", decision, fallback_used=True)

    assert diagnostics.executed_action == "hold"
    assert diagnostics.model_action == "right"
    assert diagnostics.fallback_used is True
    assert diagnostics.top_actions[0][0] == "right"
    assert len(diagnostics.top_actions) == 3
    assert diagnostics.top_actions[0][0] in API_ACTIONS
