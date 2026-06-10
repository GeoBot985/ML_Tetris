"""Compare multiple Tetris policies on the same benchmark.

Outputs a table and writes artifacts/evaluation_result.json.

Usage:
    python compare_ai.py --difficulty normal --episodes 5
    python compare_ai.py --checkpoint artifacts/ai_policy.pt --episodes 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_agent.environment import TetrisEnvironment
from ai_agent.rewards import profile_board
from ai_agent.safety import SafetyWrapper
from ai_agent.training import coach_action, load_policy_from_checkpoint
from tetris.difficulty import EASY, HARD, NORMAL

DIFFICULTIES = {"easy": EASY, "normal": NORMAL, "hard": HARD}


# ── Policy protocols ──────────────────────────────────────────────────────────

class Policy(Protocol):
    def decide(self, snapshot: dict, *, deterministic: bool = True) -> str: ...


class RandomPolicy:
    """Selects a random legal action each step."""

    _ACTIONS = ["left", "right", "soft_drop", "hard_drop", "rotate_cw", "rotate_ccw", "hold"]

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def decide(self, snapshot: dict, *, deterministic: bool = True) -> str:
        return self.rng.choice(self._ACTIONS)


class CoachPolicy:
    """Deterministic heuristic (beam-search placement scorer from training.py)."""

    def __init__(self, difficulty):
        self.difficulty = difficulty

    def decide(self, snapshot: dict, *, deterministic: bool = True) -> str:
        action, _score, _breakdown = coach_action(snapshot, difficulty=self.difficulty)
        return action


class PPOPolicyWrapper:
    """Thin wrapper making PPOPolicy conform to the Policy protocol."""

    def __init__(self, policy):
        self._policy = policy

    def decide(self, snapshot: dict, *, deterministic: bool = True) -> str:
        return self._policy.predict_from_snapshot(snapshot, deterministic=deterministic).action


class GuardedPolicyWrapper:
    """PPO policy guarded by SafetyWrapper."""

    def __init__(self, policy, difficulty):
        self._wrapper = SafetyWrapper(policy, difficulty=difficulty)
        self._corrections = 0
        self._calls = 0

    def decide(self, snapshot: dict, *, deterministic: bool = True) -> str:
        self._calls += 1
        decision = self._wrapper.decide(snapshot, deterministic=deterministic)
        if decision.corrected:
            self._corrections += 1
        return decision.executed_action

    @property
    def correction_rate(self) -> float:
        return self._corrections / self._calls if self._calls else 0.0


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class PolicyResult:
    name: str
    mean_lines: float
    mean_score: float
    game_over_rate: float
    safety_correction_rate: str


def run_policy(name: str, policy: Policy, difficulty, episodes: int, max_steps: int, seed: int) -> PolicyResult:
    env = TetrisEnvironment(difficulty=difficulty)
    total_lines = 0
    total_score = 0
    game_overs = 0

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        for _step in range(max_steps):
            snapshot = env.snapshot()
            action = policy.decide(snapshot, deterministic=True)
            _, _reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        total_lines += env.game_state.lines_cleared
        total_score += env.game_state.score
        if env.game_state.game_over:
            game_overs += 1

    correction_rate = "n/a"
    if hasattr(policy, "correction_rate"):
        correction_rate = f"{policy.correction_rate:.1%}"  # type: ignore[attr-defined]

    return PolicyResult(
        name=name,
        mean_lines=total_lines / episodes,
        mean_score=total_score / episodes,
        game_over_rate=game_overs / episodes,
        safety_correction_rate=correction_rate,
    )


# ── Formatting ────────────────────────────────────────────────────────────────

def _table(results: list[PolicyResult]) -> str:
    header = f"{'Policy':<22} {'Mean lines':>12} {'Mean score':>12} {'Game-over %':>12} {'Safety corrections':>20}"
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        rows.append(
            f"{r.name:<22} {r.mean_lines:>12.1f} {r.mean_score:>12.0f} "
            f"{r.game_over_rate:>11.0%} {r.safety_correction_rate:>20}"
        )
    return "\n".join(rows)


def _markdown_table(results: list[PolicyResult]) -> str:
    rows = [
        "| Policy | Mean lines | Mean score | Game-over % | Safety corrections |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        rows.append(
            f"| {r.name} | {r.mean_lines:.1f} | {r.mean_score:.0f} | {r.game_over_rate:.0%} | {r.safety_correction_rate} |"
        )
    return "\n".join(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_placement_policy(name: str, checkpoint: Path, difficulty, episodes: int, max_pieces: int, seed: int) -> PolicyResult:
    """Evaluate the CNN placement policy (DAgger+PPO). It plays one piece per action,
    so it uses its own PlacementEnv rather than the micro-action protocol."""
    from ai_agent.placement import PlacementEnv, load_placement_policy

    policy = load_placement_policy(checkpoint)
    env = PlacementEnv(difficulty=difficulty, max_pieces=max_pieces)
    total_lines = 0
    total_score = 0
    game_overs = 0
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        while True:
            slot = policy.act(env.game_state, deterministic=True)
            if slot is None:
                break
            step = env.step(slot)
            if step.terminated:
                break
        total_lines += env.game_state.lines_cleared
        total_score += env.game_state.score
        game_overs += int(env.game_state.game_over)
    return PolicyResult(
        name=name,
        mean_lines=total_lines / episodes,
        mean_score=total_score / episodes,
        game_over_rate=game_overs / episodes,
        safety_correction_rate="n/a",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Tetris policies side-by-side.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Legacy micro-action PPO checkpoint.")
    parser.add_argument("--placement-checkpoint", type=Path, default=None, help="CNN placement policy (DAgger+PPO) checkpoint.")
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="normal")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-pieces", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/evaluation_result.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    difficulty = DIFFICULTIES[args.difficulty]

    policies: list[tuple[str, Policy]] = [
        ("Random", RandomPolicy(args.seed)),
        ("Coach (heuristic)", CoachPolicy(difficulty)),
    ]

    if args.checkpoint and args.checkpoint.exists():
        raw_policy = load_policy_from_checkpoint(args.checkpoint)
        policies.append(("PPO", PPOPolicyWrapper(raw_policy)))
        policies.append(("Guarded PPO", GuardedPolicyWrapper(raw_policy, difficulty)))
    elif args.checkpoint:
        print(f"WARNING: checkpoint not found at {args.checkpoint} — skipping PPO policies.", file=sys.stderr)

    print(f"\nRunning {args.episodes} episodes per policy on {args.difficulty} difficulty...\n")
    results = []
    for name, policy in policies:
        print(f"  {name}...", end=" ", flush=True)
        result = run_policy(name, policy, difficulty, args.episodes, args.max_steps, args.seed)
        results.append(result)
        print(f"{result.mean_lines:.1f} lines")

    if args.placement_checkpoint and args.placement_checkpoint.exists():
        print("  CNN Placement (DAgger+PPO)...", end=" ", flush=True)
        result = run_placement_policy(
            "CNN Placement (imitation)", args.placement_checkpoint, difficulty,
            args.episodes, args.max_pieces, args.seed,
        )
        results.append(result)
        print(f"{result.mean_lines:.1f} lines")
    elif args.placement_checkpoint:
        print(f"WARNING: placement checkpoint not found at {args.placement_checkpoint}.", file=sys.stderr)

    print()
    print(_table(results))

    output = {
        "difficulty": args.difficulty,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "results": [
            {
                "policy": r.name,
                "mean_lines": round(r.mean_lines, 2),
                "mean_score": round(r.mean_score, 2),
                "game_over_rate": round(r.game_over_rate, 4),
                "safety_correction_rate": r.safety_correction_rate,
            }
            for r in results
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    report_path = args.output_json.with_name("evaluation_report.md")
    report_path.write_text(
        "\n".join(
            [
                "# AI Evaluation Report",
                "",
                f"- Difficulty: {args.difficulty}",
                f"- Episodes per policy: {args.episodes}",
                f"- Max steps: {args.max_steps}",
                f"- Seed: {args.seed}",
                "",
                _markdown_table(results),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nResults written to {args.output_json}")
    print(f"Markdown report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
