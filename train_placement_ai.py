"""Train the CNN placement policy (DAgger warm-start + PPO fine-tune).

This is the approach that actually produces a learned net that plays Tetris:
placement-level actions + a convolutional policy + DAgger imitation of the coach
+ PPO fine-tuning. See docs/ai_pipeline.md for the full rationale.

    python train_placement_ai.py --difficulty normal
    python train_placement_ai.py --dagger-iterations 5 --ppo-updates 60 --checkpoint artifacts/placement_policy.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_agent.placement import PlacementTrainConfig, evaluate_placement, load_placement_policy, train_placement_policy
from tetris.difficulty import EASY, HARD, NORMAL

DIFFICULTIES = {"easy": EASY, "normal": NORMAL, "hard": HARD}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CNN placement policy (scaled imitation of the coach).")
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="normal")
    parser.add_argument("--bc-states", type=int, default=120000, help="Coach samples for imitation — the main quality dial.")
    parser.add_argument("--bc-epochs", type=int, default=35)
    parser.add_argument("--dagger-iterations", type=int, default=0, help="Optional DAgger refinement on top of BC.")
    parser.add_argument("--ppo-updates", type=int, default=0, help="Optional PPO fine-tune (off by default; it degrades the policy here).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/placement_policy.pt"))
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/placement_training_summary.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    difficulty = DIFFICULTIES[args.difficulty]
    config = PlacementTrainConfig(
        difficulty=difficulty,
        bc_states=args.bc_states,
        bc_epochs=args.bc_epochs,
        dagger_iterations=args.dagger_iterations,
        ppo_updates=args.ppo_updates,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )
    print(f"Training CNN placement policy on {args.difficulty} difficulty...")
    summary = train_placement_policy(config, verbose=True)

    policy = load_placement_policy(args.checkpoint)
    mean_lines, game_over_rate = evaluate_placement(policy, difficulty, episodes=args.eval_episodes, seed0=12345)
    print(f"\nFinal held-out greedy: mean_lines={mean_lines:.1f} game_over_rate={game_over_rate:.2f}")
    print(f"Checkpoint: {args.checkpoint}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "difficulty": args.difficulty,
                "best_eval_lines": summary.best_eval_lines,
                "eval_lines_curve": summary.eval_lines,
                "final_mean_lines": mean_lines,
                "final_game_over_rate": game_over_rate,
                "phase_log": summary.phase_log,
                "checkpoint": str(args.checkpoint),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
