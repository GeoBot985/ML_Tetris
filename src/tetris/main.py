from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import signal
import sys
import threading

import pygame

from .api import ApiBridge, build_snapshot, start_api_server
from .difficulty import EASY, HARD, NORMAL
from .game_state import GameState
from .high_score import load_high_score, record_high_score
from .input_handler import apply_command, command_for_key
from .renderer import Renderer
from ai_agent import TrainingProgress, load_training_progress
from ai_agent.human_hints import count_human_hints, write_human_hint


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DEFAULT_CHECKPOINT = ARTIFACTS / "ai_policy.pt"
PLACEMENT_CHECKPOINT = ARTIFACTS / "placement_policy.pt"
BENCHMARK_RESULT_PATH = ARTIFACTS / "benchmark_result.json"
TRAINING_METRICS_PATH = ARTIFACTS / "training_metrics.jsonl"
TRAINING_FEEDBACK_PATH = ARTIFACTS / "training_feedback.md"
HUMAN_HINTS_PATH = ARTIFACTS / "human_hints.jsonl"
HIGH_SCORE_PATH = ARTIFACTS / "high_score.json"
START_ACTIONS = ("play", "train", "watch", "clear_ml", "evaluate", "benchmark", "quit")
ACTION_DISPLAY_NAMES = {
    "play": "Play",
    "train": "Train",
    "watch": "Watch",
    "clear_ml": "Reset AI",
    "evaluate": "Evaluate",
    "benchmark": "Benchmark",
    "quit": "Quit",
}


@dataclass
class LaunchJob:
    label: str
    command: list[str]
    process: subprocess.Popen | None = None
    thread: threading.Thread | None = None
    completed: bool = False
    stop_requested: bool = False
    returncode: int | None = None
    error: str | None = None


def create_game(selected_difficulty):
    game_state = GameState(difficulty=selected_difficulty)
    renderer = Renderer(game_state.board.width, game_state.board.height)
    return game_state, renderer


def create_window(renderer):
    return pygame.display.set_mode((renderer.surface_width, renderer.surface_height), pygame.RESIZABLE)


def snap_size_to_aspect(width: int, height: int, aspect_ratio: float) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return 1, 1
    if width / height > aspect_ratio:
        width = int(height * aspect_ratio)
    else:
        height = int(width / aspect_ratio)
    return max(1, width), max(1, height)


def present(presentation_surface, render_surface):
    window_width, window_height = presentation_surface.get_size()
    render_width, render_height = render_surface.get_size()
    scale = min(window_width / render_width, window_height / render_height)
    scaled_width = max(1, int(render_width * scale))
    scaled_height = max(1, int(render_height * scale))
    presentation_surface.fill((0, 0, 0))
    if scaled_width == render_width and scaled_height == render_height:
        scaled_surface = render_surface
    else:
        scaled_surface = pygame.transform.smoothscale(render_surface, (scaled_width, scaled_height))
    presentation_surface.blit(
        scaled_surface,
        ((window_width - scaled_width) // 2, (window_height - scaled_height) // 2),
    )


def start_playing(selected_difficulty):
    game_state, renderer = create_game(selected_difficulty)
    render_surface = pygame.Surface((renderer.surface_width, renderer.surface_height))
    window_surface = create_window(renderer)
    return game_state, renderer, render_surface, window_surface


def activate_start_action(
    selected_action: str,
    selected_difficulty,
    launch_job: LaunchJob | None,
    renderer: Renderer,
    render_surface,
    window_surface,
):
    if launch_job is not None and not launch_job.completed and selected_action != "quit":
        return True, "start", None, renderer, render_surface, window_surface, launch_job, "A launcher job is already running."
    if selected_action == "play":
        game_state, renderer, render_surface, window_surface = start_playing(selected_difficulty)
        return True, "playing", game_state, renderer, render_surface, window_surface, None, "Playing manually."
    if selected_action == "clear_ml":
        cleared = clear_ml_training_artifacts()
        message = "Cleared ML training artifacts." if cleared else "No ML training artifacts were found."
        return True, "start", None, renderer, render_surface, window_surface, launch_job, message
    if selected_action == "quit":
        return False, "start", None, renderer, render_surface, window_surface, launch_job, "Exiting launcher."

    command = build_launch_command(selected_action, selected_difficulty)
    if command is None:
        return True, "start", None, renderer, render_surface, window_surface, None, "Checkpoint missing. Train first before launching AI modes."

    launch_job = start_launch_job(selected_action, command)
    return True, "start", None, renderer, render_surface, window_surface, launch_job, f"Running {display_action_name(selected_action)}..."


def build_launch_command(action: str, selected_difficulty) -> list[str] | None:
    difficulty_name = selected_difficulty.name.lower()
    python = sys.executable
    if action == "train":
        return [
            python,
            str(ROOT / "train_ai.py"),
            "--difficulty",
            difficulty_name,
            "--parallel-envs",
            "4",
            "--use-shared-memory",
        ]
    if action == "watch":
        if not DEFAULT_CHECKPOINT.exists():
            return None
        # rotate, drop) and uses the hold piece — looks like a real player.
        return [
            python,
            str(ROOT / "run_ai.py"),
            "--render",
            "--policy",
            "ppo",
            "--checkpoint",
            str(DEFAULT_CHECKPOINT),
            "--difficulty",
            difficulty_name,
            "--quantize",
            "--fps-limit",
            "60",
            "--continuous",
            "--manifest-path",
            str(ARTIFACTS / "watch_manifest.jsonl"),
        ]
    if action == "evaluate":
        if not DEFAULT_CHECKPOINT.exists():
            return None
        return [
            python,
            str(ROOT / "evaluate_ai.py"),
            "--checkpoint",
            str(DEFAULT_CHECKPOINT),
            "--difficulty",
            difficulty_name,
            "--quantize",
            "--manifest-path",
            str(ARTIFACTS / "evaluate_manifest.jsonl"),
        ]
    if action == "benchmark":
        if not DEFAULT_CHECKPOINT.exists():
            return None
        return [
            python,
            str(ROOT / "benchmark_ai.py"),
            "--checkpoint",
            str(DEFAULT_CHECKPOINT),
            "--difficulty",
            difficulty_name,
            "--quantize",
            "--output-json",
            str(BENCHMARK_RESULT_PATH),
        ]
    return None


def clear_ml_training_artifacts() -> bool:
    cleared_any = False
    for path in (DEFAULT_CHECKPOINT, TRAINING_METRICS_PATH, TRAINING_FEEDBACK_PATH):
        if path.exists():
            path.unlink()
            cleared_any = True
    return cleared_any


def load_launcher_training_progress() -> TrainingProgress | None:
    return load_training_progress(DEFAULT_CHECKPOINT)


def record_human_hint(game_state, selected_difficulty, difficulty_order, command: str) -> bool:
    snapshot = build_snapshot("playing", game_state, selected_difficulty, difficulty_order)
    return write_human_hint(HUMAN_HINTS_PATH, snapshot, command, difficulty=selected_difficulty.name)


def start_launch_job(action: str, command: list[str]) -> LaunchJob:
    job = LaunchJob(label=action, command=command)

    try:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        job.process = subprocess.Popen(command, cwd=ROOT, creationflags=creationflags)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        job.error = str(exc)
        job.returncode = -1
        job.completed = True
        return job

    def runner() -> None:
        try:
            assert job.process is not None
            job.returncode = job.process.wait()
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            job.error = str(exc)
            job.returncode = -1
        finally:
            job.completed = True

    thread = threading.Thread(target=runner, daemon=True)
    job.thread = thread
    thread.start()
    return job


def stop_launch_job(job: LaunchJob, *, timeout: float = 5.0) -> bool:
    process = job.process
    if process is None or job.completed:
        return False

    job.stop_requested = True

    def force_stop() -> None:
        if job.completed or job.process is None:
            return
        try:
            job.process.terminate()
        except Exception:
            try:
                job.process.kill()
            except Exception:
                pass

    try:
        if sys.platform.startswith("win"):
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
    except Exception:
        force_stop()
    timer = threading.Timer(timeout, force_stop)
    timer.daemon = True
    timer.start()
    return True


def stop_all_launch_jobs(launch_jobs: list[LaunchJob], *, timeout: float = 5.0) -> None:
    for job in launch_jobs:
        if not job.completed:
            stop_launch_job(job, timeout=timeout)


def display_action_name(action: str) -> str:
    return ACTION_DISPLAY_NAMES.get(action, action.title())


def format_benchmark_status(result: dict[str, object], returncode: int | None) -> str:
    raw_mean = float(result.get("raw_mean_ms", 0.0))
    raw_p95 = float(result.get("raw_p95_ms", 0.0))
    guarded_mean = float(result.get("guarded_mean_ms", 0.0))
    guarded_p95 = float(result.get("guarded_p95_ms", 0.0))
    lines = [
        f"Benchmark finished (exit {returncode if returncode is not None else 'unknown'})",
        f"Raw mean: {raw_mean:.4f} ms | p95: {raw_p95:.4f} ms",
        f"Guarded mean: {guarded_mean:.4f} ms | p95: {guarded_p95:.4f} ms",
    ]
    return "\n".join(lines)


def apply_api_command(command, app_state, game_state, selected_difficulty):
    if command == "quit":
        return False, app_state, game_state
    if command == "start" and app_state in {"start", "game_over"}:
        return True, "playing", GameState(difficulty=selected_difficulty)
    if command == "restart":
        return True, "playing", GameState(difficulty=selected_difficulty)
    if app_state == "playing" and game_state is not None:
        apply_command(game_state, command)
        if game_state.game_over:
            return True, "game_over", game_state
    return True, app_state, game_state


def main():
    pygame.init()
    pygame.key.set_repeat(250, 60)
    selected_difficulty = NORMAL
    selected_action_index = 0
    game_state = None
    renderer = Renderer()
    window_surface = create_window(renderer)
    render_surface = pygame.Surface((renderer.surface_width, renderer.surface_height))
    pygame.display.set_caption("Tetris Clone")
    clock = pygame.time.Clock()
    elapsed = 0
    state = "start"
    difficulty_order = [EASY, NORMAL, HARD]
    status_message = "Choose a mode with Left/Right, then press Enter."
    launch_job: LaunchJob | None = None
    launch_jobs: list[LaunchJob] = []
    training_progress = load_launcher_training_progress()
    human_hint_count = count_human_hints(HUMAN_HINTS_PATH)
    high_score = load_high_score(HIGH_SCORE_PATH)
    api_bridge = ApiBridge()
    api_bridge.set_snapshot(build_snapshot(state, game_state, selected_difficulty, difficulty_order))
    api_server = start_api_server(api_bridge)
    print("AI API listening at http://127.0.0.1:8765")

    running = True
    try:
        while running:
            dt = clock.tick(60)
            if state == "playing":
                elapsed += dt
            if launch_job is not None and launch_job.completed:
                if launch_job.error is not None:
                    status_message = f"{display_action_name(launch_job.label)} failed: {launch_job.error}"
                elif launch_job.stop_requested:
                    status_message = f"{display_action_name(launch_job.label)} stopped."
                elif launch_job.label == "benchmark" and BENCHMARK_RESULT_PATH.exists():
                    try:
                        payload = json.loads(BENCHMARK_RESULT_PATH.read_text(encoding="utf-8"))
                        status_message = format_benchmark_status(payload, launch_job.returncode)
                    except Exception:
                        status_message = f"Benchmark finished (exit {launch_job.returncode})"
                else:
                    status_message = f"{display_action_name(launch_job.label)} finished (exit {launch_job.returncode})"
                if launch_job.label == "train":
                    training_progress = load_launcher_training_progress()
                if launch_job.label == "clear_ml":
                    training_progress = None
                launch_job = None
                launch_jobs[:] = [job for job in launch_jobs if not job.completed]
            for command in api_bridge.drain_commands():
                running, state, next_game_state = apply_api_command(command, state, game_state, selected_difficulty)
                if next_game_state is not game_state:
                    game_state = next_game_state
                    if game_state is not None:
                        renderer = Renderer(game_state.board.width, game_state.board.height)
                        render_surface = pygame.Surface((renderer.surface_width, renderer.surface_height))
                        window_surface = create_window(renderer)
                    elapsed = 0
                if not running:
                    break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    snapped_size = snap_size_to_aspect(event.w, event.h, renderer.surface_width / renderer.surface_height)
                    window_surface = pygame.display.set_mode(snapped_size, pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif state == "start":
                        if event.key == pygame.K_LEFT:
                            selected_action_index = (selected_action_index - 1) % len(START_ACTIONS)
                        elif event.key == pygame.K_RIGHT:
                            selected_action_index = (selected_action_index + 1) % len(START_ACTIONS)
                        elif event.key == pygame.K_s and launch_job is not None and not launch_job.completed:
                            if stop_launch_job(launch_job):
                                status_message = f"Stopping {display_action_name(launch_job.label)}..."
                        if event.key == pygame.K_RETURN:
                            selected_action = START_ACTIONS[selected_action_index]
                            running, state, next_game_state, next_renderer, next_render_surface, next_window_surface, next_launch_job, next_status_message = activate_start_action(
                                selected_action,
                                selected_difficulty,
                                launch_job,
                                renderer,
                                render_surface,
                                window_surface,
                            )
                            status_message = next_status_message
                            launch_job = next_launch_job
                            if next_launch_job is not None:
                                launch_jobs.append(next_launch_job)
                            if selected_action == "clear_ml":
                                training_progress = None
                            if next_game_state is not None:
                                game_state = next_game_state
                                renderer = next_renderer
                                render_surface = next_render_surface
                                window_surface = next_window_surface
                                elapsed = 0
                            if not running:
                                break
                        elif event.key == pygame.K_UP:
                            index = difficulty_order.index(selected_difficulty)
                            selected_difficulty = difficulty_order[(index - 1) % len(difficulty_order)]
                        elif event.key == pygame.K_DOWN:
                            index = difficulty_order.index(selected_difficulty)
                            selected_difficulty = difficulty_order[(index + 1) % len(difficulty_order)]
                    elif state == "playing" and game_state is not None:
                        command = command_for_key(event.key)
                        if command is not None:
                            if record_human_hint(game_state, selected_difficulty, difficulty_order, command):
                                human_hint_count += 1
                            apply_command(game_state, command)
                        if game_state.game_over:
                            state = "game_over"
                    elif state == "game_over":
                        if event.key in (pygame.K_RETURN, pygame.K_r):
                            game_state, renderer, render_surface, window_surface = start_playing(selected_difficulty)
                            elapsed = 0
                            state = "playing"
            if state == "playing" and game_state is not None and elapsed >= game_state.gravity_for_level():
                game_state.gravity_tick()
                elapsed = 0
            if game_state is not None and game_state.score > high_score.score:
                high_score = record_high_score(
                    HIGH_SCORE_PATH,
                    score=game_state.score,
                    lines=game_state.lines_cleared,
                    difficulty=selected_difficulty.name,
                )
            if state == "start":
                renderer.draw_start_screen(
                    render_surface,
                    selected_difficulty,
                    selected_action=START_ACTIONS[selected_action_index],
                    status_message=status_message,
                    checkpoint_ready=DEFAULT_CHECKPOINT.exists(),
                    job_active=launch_job is not None and not launch_job.completed,
                    training_progress=training_progress,
                    human_hint_count=human_hint_count,
                    high_score=high_score.score,
                )
            elif game_state is not None:
                renderer.draw(render_surface, game_state, high_score=high_score.score)
            api_bridge.set_snapshot(build_snapshot(state, game_state, selected_difficulty, difficulty_order))
            present(window_surface, render_surface)
            pygame.display.flip()
    finally:
        stop_all_launch_jobs(launch_jobs)
        api_server.shutdown()
        api_server.server_close()
        pygame.quit()


if __name__ == "__main__":
    main()
