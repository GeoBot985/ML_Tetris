from __future__ import annotations

from dataclasses import dataclass

from .policy import API_ACTIONS, PolicyDecision


@dataclass(frozen=True)
class DecisionDiagnostics:
    executed_action: str
    model_action: str
    value: float
    entropy: float
    log_prob: float
    fallback_used: bool
    correction_reason: str | None
    risk_score: float
    top_actions: tuple[tuple[str, float], ...]


def build_decision_diagnostics(
    executed_action: str,
    model_decision: PolicyDecision,
    fallback_used: bool,
    correction_reason: str | None = None,
    risk_score: float = 0.0,
) -> DecisionDiagnostics:
    logits = model_decision.logits.detach().float().cpu().squeeze(0).tolist()
    ranked = sorted(zip(API_ACTIONS, logits), key=lambda item: item[1], reverse=True)
    return DecisionDiagnostics(
        executed_action=executed_action,
        model_action=model_decision.action,
        value=float(model_decision.value.detach().cpu().item()),
        entropy=float(model_decision.entropy.detach().cpu().item()),
        log_prob=float(model_decision.log_prob.detach().cpu().item()),
        fallback_used=fallback_used,
        correction_reason=correction_reason,
        risk_score=risk_score,
        top_actions=tuple(ranked[:3]),
    )
