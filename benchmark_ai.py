from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_agent.deployment import benchmark_policy
from ai_agent.environment import TetrisEnvironment
from ai_agent.safety import SafetyWrapper
from ai_agent.training import load_policy_from_checkpoint
from tetris.difficulty import EASY, HARD, NORMAL


DIFFICULTIES = {
    "easy": EASY,
    "normal": NORMAL,
    "hard": HARD,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Tetris policy inference speed.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the policy checkpoint.")
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTIES.keys()), default="normal")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup-iterations", type=int, default=100)
    parser.add_argument("--quantize", action="store_true", help="Load the model with INT8 dynamic quantization.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional file to write benchmark results.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    env = TetrisEnvironment(difficulty=DIFFICULTIES[args.difficulty])
    policy = load_policy_from_checkpoint(args.checkpoint, quantize=args.quantize)
    controller = SafetyWrapper(policy, difficulty=DIFFICULTIES[args.difficulty])
    result = benchmark_policy(
        policy,
        env.snapshot(),
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        controller=controller,
    )

    print(f"Raw policy mean latency: {result.raw_mean_ms:.4f} ms")
    print(f"Raw policy p95 latency: {result.raw_p95_ms:.4f} ms")
    print(f"Guarded policy mean latency: {result.guarded_mean_ms:.4f} ms")
    print(f"Guarded policy p95 latency: {result.guarded_p95_ms:.4f} ms")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result.__dict__, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        default_output = Path("artifacts/benchmark_result.json")
        default_output.parent.mkdir(parents=True, exist_ok=True)
        default_output.write_text(
            json.dumps(result.__dict__, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
