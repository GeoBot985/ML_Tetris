from __future__ import annotations

import argparse
import random
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_agent.environment import TetrisEnvironment
from ai_agent.diagnostics import build_decision_diagnostics
from ai_agent.deployment import ManifestLogger, build_manifest_record
from ai_agent.policy import PPOPolicy
from ai_agent.safety import SafetyWrapper
from ai_agent.training import load_policy_from_checkpoint
from tetris.difficulty import EASY, HARD, NORMAL
from tetris.high_score import load_high_score, record_high_score
from tetris.renderer import Renderer


ARTIFACTS = ROOT / "artifacts"
HIGH_SCORE_PATH = ARTIFACTS / "high_score.json"
TRAIN_ACTIONS = ("noop", "left", "right", "soft_drop", "rotate_cw", "rotate_ccw", "hold")
_COACH_PLAY_ACTIONS = frozenset({"left", "right", "soft_drop", "hard_drop", "rotate_cw", "rotate_ccw", "hold", "noop"})
DIFFICULTIES = {
    "easy": EASY,
    "normal": NORMAL,
    "hard": HARD,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Tetris AI environment.")
    parser.add_argument("--headless", action="store_true", help="Run episodes without opening a window.")
    parser.add_argument("--render", action="store_true", help="Render the current episode with pygame.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the action sampler.")
    parser.add_argument(
        "--policy",
        choices=("random", "ppo", "placement", "coach"),
        default="random",
        help="Action source: random, legacy micro-action 'ppo', the CNN 'placement' policy, "
        "or the search-based 'coach' heuristic (plays well, no checkpoint needed).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained policy checkpoint (micro-action for --policy ppo, CNN for --policy placement).",
    )
    parser.add_argument(
        "--rows-per-second",
        type=float,
        default=14.0,
        help="Falling speed for --policy placement render (rows/sec). Lower = slower.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load the checkpoint with dynamic INT8 quantization for lower latency and memory use.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional JSONL file to record every PPO decision for auditability.",
    )
    parser.add_argument(
        "--difficulty",
        choices=sorted(DIFFICULTIES.keys()),
        default="normal",
        help="Starting difficulty for the environment.",
    )
    parser.add_argument(
        "--fps-limit",
        type=int,
        default=0,
        help="Maximum render frame rate. Use 0 for uncapped true-speed playback.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep restarting episodes in render mode until the window is closed.",
    )
    return parser


def resolve_mode(args: argparse.Namespace) -> str:
    if args.headless and args.render:
        raise ValueError("--headless and --render are mutually exclusive")
    if args.render:
        return "render"
    return "headless"


def random_action(rng: random.Random) -> str:
    return rng.choice(TRAIN_ACTIONS)


def build_action_selector(env: TetrisEnvironment, policy_name: str, seed: int | None):
    if policy_name == "ppo":
        policy = PPOPolicy.from_snapshot(env.snapshot())
        controller = SafetyWrapper(policy, difficulty=env.difficulty)

        def choose_action() -> str:
            return controller.decide(env.snapshot(), deterministic=True).executed_action

        return choose_action

    if policy_name == "coach":
        from ai_agent.training import coach_action as _coach_action
        difficulty = env.difficulty

        def choose_action() -> str:
            action, _, _ = _coach_action(env.snapshot(), difficulty)
            # coach returns "restart" on game_over; the episode terminates next step anyway
            return action if action in _COACH_PLAY_ACTIONS else "hard_drop"

        return choose_action

    rng = random.Random(seed)

    def choose_action() -> str:
        return random_action(rng)

    return choose_action


def present(pygame, presentation_surface, render_surface):
    window_width, window_height = presentation_surface.get_size()
    render_width, render_height = render_surface.get_size()
    scale = min(window_width / render_width, window_height / render_height)
    scaled_width = max(1, int(render_width * scale))
    scaled_height = max(1, int(render_height * scale))
    presentation_surface.fill((0, 0, 0))
    scaled_surface = render_surface
    if scaled_width != render_width or scaled_height != render_height:
        scaled_surface = pygame.transform.smoothscale(render_surface, (scaled_width, scaled_height))
    presentation_surface.blit(
        scaled_surface,
        ((window_width - scaled_width) // 2, (window_height - scaled_height) // 2),
    )


def _render_tick(clock, fps_limit: int) -> None:
    if fps_limit > 0:
        clock.tick(fps_limit)
    else:
        clock.tick()


def run_headless(
    env: TetrisEnvironment,
    episodes: int,
    max_steps: int,
    seed: int | None,
    policy_name: str,
) -> list[dict[str, float]]:
    choose_action = build_action_selector(env, policy_name, seed)
    summaries = []
    for episode in range(1, episodes + 1):
        _, info = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        while steps < max_steps and not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(choose_action())
            total_reward += reward
            steps += 1
        summaries.append(
            {
                "episode": float(episode),
                "steps": float(steps),
                "reward": float(total_reward),
                "game_over": float(bool(env.game_state.game_over)),
            }
        )
        print(
            f"episode={episode} steps={steps} reward={total_reward:.2f} "
            f"lines={env.game_state.lines_cleared} score={env.game_state.score}"
        )
    return summaries


def run_render(
    env: TetrisEnvironment,
    episodes: int,
    max_steps: int,
    seed: int | None,
    policy_name: str,
    fps_limit: int,
    continuous: bool,
) -> list[dict[str, float]]:
    import pygame

    pygame.init()
    try:
        choose_action = build_action_selector(env, policy_name, seed)
        renderer = Renderer(env.game_state.board.width, env.game_state.board.height)
        window_surface = pygame.display.set_mode((renderer.surface_width, renderer.surface_height), pygame.RESIZABLE)
        render_surface = pygame.Surface((renderer.surface_width, renderer.surface_height))
        clock = pygame.time.Clock()
        summaries = []
        high_score = load_high_score(HIGH_SCORE_PATH)
        episode = 1
        while continuous or episode <= episodes:
            episode_seed = None if seed is None else seed + episode - 1
            env.reset(seed=episode_seed)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            while steps < max_steps and not (terminated or truncated):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return summaries
                _, reward, terminated, truncated, _ = env.step(choose_action())
                if env.game_state.score > high_score.score:
                    high_score = record_high_score(
                        HIGH_SCORE_PATH,
                        score=env.game_state.score,
                        lines=env.game_state.lines_cleared,
                        difficulty=env.difficulty.name,
                    )
                total_reward += reward
                steps += 1
                render_surface.fill((18, 18, 24))
                renderer.draw(render_surface, env.game_state, high_score=high_score.score)
                present(pygame, window_surface, render_surface)
                pygame.display.flip()
                _render_tick(clock, fps_limit)
            summaries.append(
                {
                    "episode": float(episode),
                    "steps": float(steps),
                    "reward": float(total_reward),
                    "game_over": float(bool(env.game_state.game_over)),
                }
            )
            print(
                f"episode={episode} steps={steps} reward={total_reward:.2f} "
                f"lines={env.game_state.lines_cleared} score={env.game_state.score}"
            )
            episode += 1
        return summaries
    finally:
        pygame.quit()


def run_render_placement(
    difficulty,
    checkpoint: Path | None,
    episodes: int,
    seed: int | None,
    rows_per_second: float,
    continuous: bool,
    use_coach: bool = False,
) -> list[dict[str, float]]:
    """Render a placement-level agent in a pygame window with a falling animation.

    Two action sources share this loop because both decide one whole piece-drop at a
    time: the CNN ``placement`` policy (from ``checkpoint``) and the search-based
    ``coach`` heuristic (``use_coach``). Each chosen placement is animated — the piece
    appears in its target column at the top and falls one row per frame to its landing
    spot, then locks.
    """
    import pygame

    from ai_agent.placement import PlacementEnv, coach_slot, legal_placements, load_placement_policy
    from tetris.renderer import Renderer

    policy = None if use_coach else load_placement_policy(checkpoint)
    env = PlacementEnv(difficulty=difficulty, max_pieces=10_000)

    def choose_slot(game_state):
        if use_coach:
            return coach_slot(game_state, difficulty)
        return policy.act(game_state, deterministic=True)

    pygame.init()
    try:
        renderer = Renderer(env.game_state.board.width, env.game_state.board.height)
        window_surface = pygame.display.set_mode((renderer.surface_width, renderer.surface_height), pygame.RESIZABLE)
        render_surface = pygame.Surface((renderer.surface_width, renderer.surface_height))
        clock = pygame.time.Clock()
        rows_per_second = max(1.0, rows_per_second)
        summaries: list[dict[str, float]] = []
        high_score = load_high_score(HIGH_SCORE_PATH)
        episode = 1

        def paint() -> bool:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
            render_surface.fill((18, 18, 24))
            renderer.draw(render_surface, env.game_state, high_score=high_score.score)
            present(pygame, window_surface, render_surface)
            pygame.display.flip()
            clock.tick(60)
            return True

        running = True
        while running and (continuous or episode <= episodes):
            env.reset(seed=None if seed is None else seed + episode - 1)
            gs = env.game_state
            while not gs.game_over:
                placements = legal_placements(gs)
                if not placements:
                    break
                slot = choose_slot(gs)
                if slot is None or slot not in placements:
                    break
                target = placements[slot]
                # Place the rotated piece at its target column, top of the board,
                # then let it fall one row per tick to the landing position.
                gs.active_piece = target.piece
                gs.active_x = target.x
                gs.active_y = 0
                frame_delay_ms = 1000.0 / rows_per_second
                last = pygame.time.get_ticks()
                while gs.active_y < target.y:
                    if not paint():
                        running = False
                        break
                    now = pygame.time.get_ticks()
                    if now - last >= frame_delay_ms:
                        gs.active_y += 1
                        last = now
                if not running:
                    break
                gs.active_y = target.y
                gs.lock_active_piece()  # clears lines, scores, spawns next piece
                if gs.score > high_score.score:
                    high_score = record_high_score(
                        HIGH_SCORE_PATH,
                        score=gs.score,
                        lines=gs.lines_cleared,
                        difficulty=difficulty.name,
                    )
                if not paint():
                    running = False
                    break
                pygame.time.delay(60)  # brief settle pause between pieces
            summaries.append(
                {
                    "episode": float(episode),
                    "lines": float(gs.lines_cleared),
                    "score": float(gs.score),
                    "game_over": float(bool(gs.game_over)),
                }
            )
            print(f"episode={episode} lines={gs.lines_cleared} score={gs.score} game_over={gs.game_over}")
            # Pause on the game-over frame so it is visible before restart.
            if running:
                pygame.time.delay(900)
            episode += 1
        return summaries
    finally:
        pygame.quit()


def build_policy_for_checkpoint(env: TetrisEnvironment, checkpoint: Path | None):
    if checkpoint is None:
        return PPOPolicy.from_snapshot(env.snapshot())
    return load_policy_from_checkpoint(checkpoint)


def build_hybrid_controller(
    env: TetrisEnvironment,
    checkpoint: Path | None,
    difficulty,
    *,
    quantize: bool = False,
) -> SafetyWrapper:
    if checkpoint is None:
        policy = PPOPolicy.from_snapshot(env.snapshot())
    else:
        policy = load_policy_from_checkpoint(checkpoint, quantize=quantize)
    return SafetyWrapper(policy, difficulty=difficulty)


def run_headless_with_policy(
    env: TetrisEnvironment,
    controller: SafetyWrapper,
    episodes: int,
    max_steps: int,
    seed: int | None,
    manifest_logger: ManifestLogger | None = None,
) -> list[dict[str, float]]:
    summaries = []
    for episode in range(1, episodes + 1):
        env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        while steps < max_steps and not (terminated or truncated):
            snapshot = env.snapshot()
            start_ns = time.perf_counter_ns()
            decision = controller.decide(snapshot, deterministic=True)
            latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            _, reward, terminated, truncated, _ = env.step(decision.executed_action)
            if manifest_logger is not None:
                manifest_logger.record(
                    build_manifest_record(
                        run_id=manifest_logger.run_id or "run",
                        episode=episode,
                        step=steps,
                        snapshot=snapshot,
                        safety_decision=decision,
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
                        diagnostics=None,
                        latency_ms=latency_ms,
                    )
                )
            total_reward += reward
            steps += 1
        summaries.append(
            {
                "episode": float(episode),
                "steps": float(steps),
                "reward": float(total_reward),
                "game_over": float(bool(env.game_state.game_over)),
            }
        )
        print(
            f"episode={episode} steps={steps} reward={total_reward:.2f} "
            f"lines={env.game_state.lines_cleared} score={env.game_state.score}"
        )
        if manifest_logger is not None:
            manifest_logger.record_event(
                "episode_summary",
                {
                    "run_id": manifest_logger.run_id or "run",
                    "episode": episode,
                    "steps": steps,
                    "reward": total_reward,
                    "lines": env.game_state.lines_cleared,
                    "score": env.game_state.score,
                    "game_over": bool(env.game_state.game_over),
                },
            )
    return summaries


def run_render_with_policy(
    env: TetrisEnvironment,
    controller: SafetyWrapper,
    episodes: int,
    max_steps: int,
    seed: int | None,
    fps_limit: int,
    continuous: bool,
    manifest_logger: ManifestLogger | None = None,
) -> list[dict[str, float]]:
    import pygame

    pygame.init()
    try:
        renderer = Renderer(env.game_state.board.width, env.game_state.board.height)
        window_surface = pygame.display.set_mode((renderer.surface_width, renderer.surface_height), pygame.RESIZABLE)
        render_surface = pygame.Surface((renderer.surface_width, renderer.surface_height))
        clock = pygame.time.Clock()
        summaries = []
        high_score = load_high_score(HIGH_SCORE_PATH)
        episode = 1
        while continuous or episode <= episodes:
            episode_seed = None if seed is None else seed + episode - 1
            env.reset(seed=episode_seed)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            while steps < max_steps and not (terminated or truncated):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return summaries
                snapshot = env.snapshot()
                start_ns = time.perf_counter_ns()
                decision = controller.decide(snapshot, deterministic=True)
                latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                diagnostics = build_decision_diagnostics(
                    executed_action=decision.executed_action,
                    model_decision=decision.model_decision,
                    fallback_used=decision.corrected,
                    correction_reason=decision.correction_reason,
                    risk_score=decision.risk_score,
                )
                _, reward, terminated, truncated, _ = env.step(decision.executed_action)
                if env.game_state.score > high_score.score:
                    high_score = record_high_score(
                        HIGH_SCORE_PATH,
                        score=env.game_state.score,
                        lines=env.game_state.lines_cleared,
                        difficulty=env.difficulty.name,
                    )
                if manifest_logger is not None:
                    manifest_logger.record(
                        build_manifest_record(
                            run_id=manifest_logger.run_id or "run",
                            episode=episode,
                            step=steps,
                            snapshot=snapshot,
                            safety_decision=decision,
                            reward=float(reward),
                            terminated=terminated,
                            truncated=truncated,
                            diagnostics=diagnostics,
                            latency_ms=latency_ms,
                        )
                    )
                total_reward += reward
                steps += 1
                render_surface.fill((18, 18, 24))
                renderer.draw(render_surface, env.game_state, diagnostics=diagnostics, high_score=high_score.score)
                present(pygame, window_surface, render_surface)
                pygame.display.flip()
                _render_tick(clock, fps_limit)
            summaries.append(
                {
                    "episode": float(episode),
                    "steps": float(steps),
                    "reward": float(total_reward),
                    "game_over": float(bool(env.game_state.game_over)),
                }
            )
            print(
                f"episode={episode} steps={steps} reward={total_reward:.2f} "
                f"lines={env.game_state.lines_cleared} score={env.game_state.score}"
            )
            if manifest_logger is not None:
                manifest_logger.record_event(
                    "episode_summary",
                    {
                        "run_id": manifest_logger.run_id or "run",
                        "episode": episode,
                        "steps": steps,
                        "reward": total_reward,
                        "lines": env.game_state.lines_cleared,
                        "score": env.game_state.score,
                        "game_over": bool(env.game_state.game_over),
                    },
                )
            episode += 1
        return summaries
    finally:
        pygame.quit()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = resolve_mode(args)

    if args.policy == "placement":
        if mode != "render":
            parser.error("--policy placement is only supported with --render")
        if args.checkpoint is None:
            parser.error("--policy placement requires --checkpoint (e.g. artifacts/placement_policy.pt)")
        run_render_placement(
            DIFFICULTIES[args.difficulty],
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            seed=args.seed,
            rows_per_second=args.rows_per_second,
            continuous=args.continuous,
            use_coach=False,
        )
        return 0

    env = TetrisEnvironment(difficulty=DIFFICULTIES[args.difficulty])
    # Coach needs many steps per episode (every key-press is one step).
    max_steps = 20_000 if args.policy == "coach" else args.max_steps
    manifest_logger = None
    try:
        if args.policy == "ppo":
            controller = build_hybrid_controller(
                env,
                args.checkpoint,
                DIFFICULTIES[args.difficulty],
                quantize=args.quantize,
            )
            if args.manifest_path is not None:
                manifest_logger = ManifestLogger(
                    args.manifest_path,
                    metadata={
                        "run_id": f"run-{int(time.time())}",
                        "mode": mode,
                        "policy": args.policy,
                        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
                        "difficulty": args.difficulty,
                        "quantized": bool(args.quantize),
                        "seed": args.seed,
                    },
                )
            if mode == "render":
                run_render_with_policy(
                    env,
                    controller,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    seed=args.seed,
                    fps_limit=args.fps_limit,
                    continuous=args.continuous,
                    manifest_logger=manifest_logger,
                )
            else:
                run_headless_with_policy(
                    env,
                    controller,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    seed=args.seed,
                    manifest_logger=manifest_logger,
                )
        elif mode == "render":
            run_render(
                env,
                episodes=args.episodes,
                max_steps=max_steps,
                seed=args.seed,
                policy_name=args.policy,
                fps_limit=args.fps_limit,
                continuous=args.continuous,
            )
        else:
            run_headless(env, episodes=args.episodes, max_steps=max_steps, seed=args.seed, policy_name=args.policy)
    finally:
        if manifest_logger is not None:
            manifest_logger.summarize(
                {
                    "run_id": manifest_logger.run_id or "run",
                    "episodes": args.episodes,
                    "max_steps": args.max_steps,
                    "difficulty": args.difficulty,
                    "quantized": bool(args.quantize),
                }
            )
            manifest_logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
