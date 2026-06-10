from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_agent.evaluation import evaluate_policy, format_evaluation_report, load_human_baselines, result_to_dict
from ai_agent.deployment import ManifestLogger, build_manifest_record
from ai_agent.training import load_policy_from_checkpoint
from ai_agent.safety import SafetyWrapper
from tetris.difficulty import EASY, HARD, NORMAL


DIFFICULTIES = {
    "easy": EASY,
    "normal": NORMAL,
    "hard": HARD,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Tetris policy against a human baseline.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the policy checkpoint.")
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTIES.keys()), default="normal")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--baseline-file", type=Path, default=None, help="Optional JSON file with human baseline scores.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional file to write the structured report.")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load the checkpoint with dynamic INT8 quantization before evaluation.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional JSONL file to record every evaluated decision.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    baselines = load_human_baselines(args.baseline_file)
    baseline = baselines[args.difficulty]
    policy = load_policy_from_checkpoint(args.checkpoint, quantize=args.quantize)
    controller = SafetyWrapper(policy, difficulty=DIFFICULTIES[args.difficulty])
    manifest_logger = None
    result = None
    try:
        if args.manifest_path is not None:
            manifest_logger = ManifestLogger(
                args.manifest_path,
                metadata={
                    "run_id": f"eval-{int(time.time())}",
                    "checkpoint": str(args.checkpoint),
                    "difficulty": args.difficulty,
                    "quantized": bool(args.quantize),
                    "episodes": args.episodes,
                    "max_steps": args.max_steps,
                    "seed": args.seed,
                },
            )
        result = evaluate_policy(
            policy,
            DIFFICULTIES[args.difficulty],
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            baseline=baseline,
            controller=controller,
            manifest_logger=manifest_logger,
        )
        report = format_evaluation_report(result)
        print(report)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result_to_dict(result), indent=2, sort_keys=True), encoding="utf-8")
    finally:
        if manifest_logger is not None and result is not None:
            manifest_logger.summarize(
                {
                    "run_id": manifest_logger.run_id or "run",
                    "episodes": args.episodes,
                    "max_steps": args.max_steps,
                    "difficulty": args.difficulty,
                    "mean_lines": result.mean_lines,
                    "mean_score": result.mean_score,
                    "clears_baseline": result.clears_baseline,
                }
            )
            manifest_logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
