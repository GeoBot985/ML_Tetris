from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from .environment import TetrisEnvironment
from .deployment import ManifestLogger, build_manifest_record
from .policy import PPOPolicy
from .safety import SafetyWrapper


@dataclass(frozen=True)
class HumanBaseline:
    label: str
    difficulty: str
    mean_lines: float
    mean_score: float


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    max_steps: int
    mean_lines: float
    mean_score: float
    best_lines: int
    best_score: int
    baseline: HumanBaseline

    @property
    def lines_delta(self) -> float:
        return self.mean_lines - self.baseline.mean_lines

    @property
    def score_delta(self) -> float:
        return self.mean_score - self.baseline.mean_score

    @property
    def clears_baseline(self) -> bool:
        return self.mean_lines >= self.baseline.mean_lines and self.mean_score >= self.baseline.mean_score


DEFAULT_HUMAN_BASELINES = {
    "easy": HumanBaseline(label="Human baseline", difficulty="easy", mean_lines=50.0, mean_score=16000.0),
    "normal": HumanBaseline(label="Human baseline", difficulty="normal", mean_lines=40.0, mean_score=12000.0),
    "hard": HumanBaseline(label="Human baseline", difficulty="hard", mean_lines=28.0, mean_score=9000.0),
}


def load_human_baselines(path: Path | None = None) -> dict[str, HumanBaseline]:
    if path is None or not path.exists():
        return dict(DEFAULT_HUMAN_BASELINES)
    payload = json.loads(path.read_text(encoding="utf-8"))
    baselines = {}
    for difficulty, entry in payload.items():
        baselines[difficulty] = HumanBaseline(
            label=entry.get("label", "Human baseline"),
            difficulty=difficulty,
            mean_lines=float(entry["mean_lines"]),
            mean_score=float(entry["mean_score"]),
        )
    return baselines


def save_human_baselines(path: Path, baselines: dict[str, HumanBaseline]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        difficulty: {
            "label": baseline.label,
            "mean_lines": baseline.mean_lines,
            "mean_score": baseline.mean_score,
        }
        for difficulty, baseline in baselines.items()
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def evaluate_policy(
    policy: PPOPolicy,
    difficulty,
    *,
    episodes: int = 3,
    max_steps: int = 700,
    seed: int | None = 1,
    baseline: HumanBaseline | None = None,
    controller: SafetyWrapper | None = None,
    manifest_logger: ManifestLogger | None = None,
) -> EvaluationResult:
    env = TetrisEnvironment(difficulty=difficulty)
    total_lines = 0
    total_score = 0
    best_lines = 0
    best_score = 0

    for episode in range(episodes):
        env.reset(seed=None if seed is None else seed + episode)
        for step in range(max_steps):
            snapshot = env.snapshot()
            if controller is not None:
                decision = controller.decide(snapshot, deterministic=True)
                action = decision.executed_action
            else:
                decision = policy.act_from_snapshot(snapshot, deterministic=True)
                action = decision.action
            _, reward, terminated, truncated, _ = env.step(action)
            if manifest_logger is not None and controller is not None:
                manifest_logger.record(
                    build_manifest_record(
                        run_id=manifest_logger.run_id or "run",
                        episode=episode + 1,
                        step=step,
                        snapshot=snapshot,
                        safety_decision=decision,
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
                        diagnostics=None,
                    )
                )
            if terminated or truncated:
                break
        total_lines += env.game_state.lines_cleared
        total_score += env.game_state.score
        best_lines = max(best_lines, env.game_state.lines_cleared)
        best_score = max(best_score, env.game_state.score)
        if manifest_logger is not None:
            manifest_logger.record_event(
                "episode_summary",
                {
                    "run_id": manifest_logger.run_id or "run",
                    "episode": episode + 1,
                    "lines": env.game_state.lines_cleared,
                    "score": env.game_state.score,
                    "game_over": bool(env.game_state.game_over),
                },
            )

    baseline = baseline or DEFAULT_HUMAN_BASELINES["normal"]
    return EvaluationResult(
        episodes=episodes,
        max_steps=max_steps,
        mean_lines=total_lines / episodes if episodes else 0.0,
        mean_score=total_score / episodes if episodes else 0.0,
        best_lines=best_lines,
        best_score=best_score,
        baseline=baseline,
    )


def format_evaluation_report(result: EvaluationResult) -> str:
    status = "beats" if result.clears_baseline else "does not beat"
    return "\n".join(
        [
            f"Baseline: {result.baseline.label} ({result.baseline.difficulty})",
            f"Agent mean lines: {result.mean_lines:.2f} vs baseline {result.baseline.mean_lines:.2f} ({result.lines_delta:+.2f})",
            f"Agent mean score: {result.mean_score:.2f} vs baseline {result.baseline.mean_score:.2f} ({result.score_delta:+.2f})",
            f"Best episode: {result.best_lines} lines, {result.best_score} score",
            f"Result: agent {status} the configured human baseline",
        ]
    )


def result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    return {
        "episodes": result.episodes,
        "max_steps": result.max_steps,
        "mean_lines": result.mean_lines,
        "mean_score": result.mean_score,
        "best_lines": result.best_lines,
        "best_score": result.best_score,
        "baseline": {
            "label": result.baseline.label,
            "difficulty": result.baseline.difficulty,
            "mean_lines": result.baseline.mean_lines,
            "mean_score": result.baseline.mean_score,
        },
        "clears_baseline": result.clears_baseline,
    }
