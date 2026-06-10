from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_agent.training import TrainingConfig, train_policy
from tetris.difficulty import EASY, HARD, NORMAL


DIFFICULTIES = {
    "easy": EASY,
    "normal": NORMAL,
    "hard": HARD,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Tetris policy headlessly.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--piece-source",
        choices=("classic_uniform", "seven_bag"),
        default="classic_uniform",
        help="Training piece generator. classic_uniform is memoryless; seven_bag is modern bag randomization.",
    )
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTIES.keys()), default="normal")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/ai_policy.pt"))
    parser.add_argument("--log-path", type=Path, default=Path("artifacts/training_metrics.jsonl"))
    parser.add_argument("--feedback-path", type=Path, default=Path("artifacts/training_feedback.md"))
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument("--improvement-lines-target", type=int, default=50)
    parser.add_argument("--parallel-envs", type=int, default=1, help="Number of parallel training environments to run.")
    parser.add_argument("--use-shared-memory", action="store_true", help="Back vectorized env buffers with shared memory.")
    parser.add_argument("--human-hints", type=Path, default=Path("artifacts/human_hints.jsonl"))
    parser.add_argument("--human-hint-weight", type=float, default=0.05)
    parser.add_argument("--human-hint-decay", type=float, default=0.995)
    parser.add_argument("--human-hint-batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-quantize-verification",
        action="store_true",
        help="Skip the post-training quantized checkpoint verification step.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = TrainingConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        clip_range=args.clip_range,
        gae_lambda=args.gae_lambda,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        entropy_weight=args.entropy_weight,
        value_weight=args.value_weight,
        seed=args.seed,
        piece_source=args.piece_source,
        difficulty=DIFFICULTIES[args.difficulty],
        checkpoint_path=args.checkpoint,
        log_path=args.log_path,
        feedback_path=args.feedback_path,
        evaluation_interval=args.evaluation_interval,
        evaluation_episodes=args.evaluation_episodes,
        improvement_lines_target=args.improvement_lines_target,
        parallel_envs=args.parallel_envs,
        use_shared_memory=args.use_shared_memory,
        verify_quantized_checkpoint=not args.skip_quantize_verification,
        human_hints_path=args.human_hints,
        human_hint_weight=args.human_hint_weight,
        human_hint_decay=args.human_hint_decay,
        human_hint_batch_size=args.human_hint_batch_size,
    )

    summary = train_policy(config)
    print(
        f"Training complete: episodes={len(summary.episodes)} "
        f"best_lines={summary.best_lines} best_reward={summary.best_reward:.2f} "
        f"checkpoint={summary.checkpoint_path}"
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
