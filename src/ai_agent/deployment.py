from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import time
from pathlib import Path
from typing import Any

import torch

from .environment import TetrisEnvironment
from .policy import PPOPolicy
from .safety import SafetyDecision, SafetyWrapper

try:
    from torch.ao.quantization import quantize_dynamic
except ImportError:  # pragma: no cover - older torch fallback
    from torch.quantization import quantize_dynamic  # type: ignore[no-redef]


@dataclass(frozen=True)
class BenchmarkResult:
    iterations: int
    warmup_iterations: int
    raw_mean_ms: float
    raw_p95_ms: float
    guarded_mean_ms: float
    guarded_p95_ms: float


def quantize_policy(policy: PPOPolicy) -> PPOPolicy:
    if policy.device.type != "cpu":
        raise ValueError("dynamic INT8 quantization is supported on CPU policies only")

    quantized_model = quantize_dynamic(policy.model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)
    return PPOPolicy(quantized_model, device="cpu", fallback_action_fn=policy.fallback_action_fn)


def load_policy_for_deployment(
    policy: PPOPolicy,
    *,
    quantize: bool = False,
) -> PPOPolicy:
    if not quantize:
        return policy
    return quantize_policy(policy)


def benchmark_policy(
    policy: PPOPolicy,
    snapshot: dict[str, Any],
    *,
    iterations: int = 1000,
    warmup_iterations: int = 100,
    controller: SafetyWrapper | None = None,
) -> BenchmarkResult:
    raw_samples: list[float] = []
    guarded_samples: list[float] = []

    with torch.inference_mode():
        for i in range(warmup_iterations + iterations):
            start = time.perf_counter()
            policy.predict_from_snapshot(snapshot, deterministic=True)
            elapsed = (time.perf_counter() - start) * 1000.0
            if i >= warmup_iterations:
                raw_samples.append(elapsed)

        guarded_controller = controller or SafetyWrapper(policy, difficulty=TetrisEnvironment().difficulty)
        for i in range(warmup_iterations + iterations):
            start = time.perf_counter()
            guarded_controller.decide(snapshot, deterministic=True)
            elapsed = (time.perf_counter() - start) * 1000.0
            if i >= warmup_iterations:
                guarded_samples.append(elapsed)

    raw_samples.sort()
    guarded_samples.sort()
    raw_mean = sum(raw_samples) / len(raw_samples) if raw_samples else 0.0
    guarded_mean = sum(guarded_samples) / len(guarded_samples) if guarded_samples else 0.0
    raw_p95 = raw_samples[int(len(raw_samples) * 0.95) - 1] if raw_samples else 0.0
    guarded_p95 = guarded_samples[int(len(guarded_samples) * 0.95) - 1] if guarded_samples else 0.0
    return BenchmarkResult(
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        raw_mean_ms=raw_mean,
        raw_p95_ms=raw_p95,
        guarded_mean_ms=guarded_mean,
        guarded_p95_ms=guarded_p95,
    )


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def build_manifest_record(
    *,
    run_id: str,
    episode: int,
    step: int,
    snapshot: dict[str, Any],
    safety_decision: SafetyDecision,
    reward: float,
    terminated: bool,
    truncated: bool,
    diagnostics: Any | None = None,
    latency_ms: float | None = None,
    source: str = "ppo",
) -> dict[str, Any]:
    diagnostics_payload = None
    if diagnostics is not None:
        diagnostics_payload = {
            "executed_action": diagnostics.executed_action,
            "model_action": diagnostics.model_action,
            "value": diagnostics.value,
            "entropy": diagnostics.entropy,
            "log_prob": diagnostics.log_prob,
            "fallback_used": diagnostics.fallback_used,
            "correction_reason": diagnostics.correction_reason,
            "risk_score": diagnostics.risk_score,
            "top_actions": list(diagnostics.top_actions),
        }

    return {
        "type": "decision",
        "run_id": run_id,
        "episode": episode,
        "step": step,
        "source": source,
        "state_hash": snapshot_digest(snapshot),
        "snapshot": snapshot,
        "model_action": safety_decision.model_decision.action,
        "executed_action": safety_decision.executed_action,
        "action_index": safety_decision.model_decision.action_index,
        "corrected": safety_decision.corrected,
        "correction_reason": safety_decision.correction_reason,
        "risk_score": safety_decision.risk_score,
        "legal_action": safety_decision.legal_action,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "latency_ms": latency_ms,
        "diagnostics": diagnostics_payload,
    }


class ManifestLogger:
    def __init__(self, path: Path, *, metadata: dict[str, Any] | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._closed = False
        self.run_id = metadata.get("run_id") if metadata else None
        self.metadata = metadata or {}
        self._write_event("run_start", self.metadata)

    def _write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {"type": event_type, **payload}
        self._file.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._file.flush()

    def record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("ManifestLogger is closed")
        self._write_event(event_type, payload)

    def record(self, payload: dict[str, Any]) -> None:
        self.record_event("decision", payload)

    def summarize(self, payload: dict[str, Any]) -> None:
        self.record_event("run_summary", payload)

    def close(self) -> None:
        if self._closed:
            return
        self._file.close()
        self._closed = True
