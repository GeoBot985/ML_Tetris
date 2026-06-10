from __future__ import annotations

from pathlib import Path
import signal
import threading

import pygame
import torch

from tetris.difficulty import NORMAL
import tetris.main as tetris_main
from tetris.main import (
    LaunchJob,
    START_ACTIONS,
    activate_start_action,
    build_launch_command,
    format_benchmark_status,
    start_launch_job,
    stop_launch_job,
    stop_all_launch_jobs,
)
from tetris.renderer import Renderer


def test_build_launch_command_for_train_uses_local_script():
    command = build_launch_command("train", NORMAL)

    assert command is not None
    assert command[0].endswith("python.exe") or command[0].endswith("python")
    assert command[1].endswith("train_ai.py")
    assert "--difficulty" in command


def test_build_launch_command_requires_checkpoint_for_ai_modes():
    original_checkpoint = tetris_main.DEFAULT_CHECKPOINT
    tetris_main.DEFAULT_CHECKPOINT = Path("artifacts/__missing_checkpoint__.pt")
    try:
        assert build_launch_command("watch", NORMAL) is None
        assert build_launch_command("evaluate", NORMAL) is None
        assert build_launch_command("benchmark", NORMAL) is None
    finally:
        tetris_main.DEFAULT_CHECKPOINT = original_checkpoint


def test_build_launch_command_watch_uses_true_speed_rendering(tmp_path):
    original_checkpoint = tetris_main.DEFAULT_CHECKPOINT
    checkpoint = tmp_path / "ai_policy.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    tetris_main.DEFAULT_CHECKPOINT = checkpoint
    try:
        command = build_launch_command("watch", NORMAL)
        assert command is not None
        assert command[command.index("--policy") + 1] == "ppo"
        assert "--checkpoint" in command
        assert command[command.index("--checkpoint") + 1].endswith("ai_policy.pt")
        assert "--quantize" in command
        assert "--fps-limit" in command
        assert command[command.index("--fps-limit") + 1] == "60"
        assert "--continuous" in command
    finally:
        tetris_main.DEFAULT_CHECKPOINT = original_checkpoint


def test_build_launch_command_benchmark_writes_json(tmp_path):
    original_checkpoint = tetris_main.DEFAULT_CHECKPOINT
    checkpoint = tmp_path / "ai_policy.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    tetris_main.DEFAULT_CHECKPOINT = checkpoint
    try:
        command = build_launch_command("benchmark", NORMAL)
        assert command is not None
        assert "--output-json" in command
        assert command[command.index("--output-json") + 1].endswith("benchmark_result.json")
    finally:
        tetris_main.DEFAULT_CHECKPOINT = original_checkpoint


def test_format_benchmark_status_includes_latency_metrics():
    status = format_benchmark_status(
        {
            "raw_mean_ms": 0.1234,
            "raw_p95_ms": 0.2345,
            "guarded_mean_ms": 0.3456,
            "guarded_p95_ms": 0.4567,
        },
        0,
    )

    assert "Benchmark finished (exit 0)" in status
    assert "Raw mean: 0.1234 ms" in status
    assert "Guarded mean: 0.3456 ms" in status


def test_activate_start_action_launches_each_mode(monkeypatch, tmp_path):
    pygame.init()
    try:
        renderer = Renderer()
        surface = pygame.Surface((renderer.surface_width, renderer.surface_height))
        window_surface = pygame.Surface((renderer.surface_width, renderer.surface_height))

        launched = []

        def fake_start_playing(selected_difficulty):
            return "game_state", "play_renderer", "play_surface", "play_window"

        def fake_start_launch_job(action, command):
            launched.append((action, command))
            return LaunchJob(label=action, command=command)

        checkpoint = tmp_path / "ai_policy.pt"
        checkpoint.write_text("stub", encoding="utf-8")

        monkeypatch.setattr(tetris_main, "start_playing", fake_start_playing)
        monkeypatch.setattr(tetris_main, "start_launch_job", fake_start_launch_job)
        monkeypatch.setattr(tetris_main, "DEFAULT_CHECKPOINT", checkpoint)

        running, state, game_state, new_renderer, new_surface, new_window, launch_job, message = activate_start_action(
            "play",
            NORMAL,
            None,
            renderer,
            surface,
            window_surface,
        )
        assert running is True
        assert state == "playing"
        assert game_state == "game_state"
        assert launch_job is None
        assert message == "Playing manually."

        for action in ("train", "watch", "evaluate", "benchmark"):
            launched.clear()
            running, state, game_state, new_renderer, new_surface, new_window, launch_job, message = activate_start_action(
                action,
                NORMAL,
                None,
                renderer,
                surface,
                window_surface,
            )
            assert running is True
            assert state == "start"
            assert game_state is None
            assert launch_job is not None
            assert launch_job.label == action
            assert launched, action
            assert launched[0][0] == action
            assert launched[0][1][0].endswith("python.exe") or launched[0][1][0].endswith("python")
            assert message.startswith("Running ")

        running, state, game_state, new_renderer, new_surface, new_window, launch_job, message = activate_start_action(
            "clear_ml",
            NORMAL,
            None,
            renderer,
            surface,
            window_surface,
        )
        assert running is True
        assert state == "start"
        assert "Cleared ML training artifacts" in message or "No ML training artifacts were found" in message

        running, state, game_state, new_renderer, new_surface, new_window, launch_job, message = activate_start_action(
            "quit",
            NORMAL,
            None,
            renderer,
            surface,
            window_surface,
        )
        assert running is False
        assert state == "start"
        assert message == "Exiting launcher."
    finally:
        pygame.quit()


def test_clear_ml_training_artifacts_removes_training_files(monkeypatch, tmp_path):
    checkpoint = tmp_path / "ai_policy.pt"
    metrics = tmp_path / "training_metrics.jsonl"
    feedback = tmp_path / "training_feedback.md"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    metrics.write_text("metrics", encoding="utf-8")
    feedback.write_text("feedback", encoding="utf-8")

    monkeypatch.setattr(tetris_main, "DEFAULT_CHECKPOINT", checkpoint)
    monkeypatch.setattr(tetris_main, "TRAINING_METRICS_PATH", metrics)
    monkeypatch.setattr(tetris_main, "TRAINING_FEEDBACK_PATH", feedback)

    cleared = tetris_main.clear_ml_training_artifacts()

    assert cleared is True
    assert not checkpoint.exists()
    assert not metrics.exists()
    assert not feedback.exists()


def test_load_launcher_training_progress_reads_checkpoint_summary(tmp_path, monkeypatch):
    checkpoint = tmp_path / "ai_policy.pt"
    torch.save(
        {
            "summary": {
                "best_lines": 42,
                "best_reward": 123.5,
                "episodes": [
                    {"steps": 100},
                    {"steps": 150},
                    {"steps": 250},
                ],
            }
        },
        checkpoint,
    )
    monkeypatch.setattr(tetris_main, "DEFAULT_CHECKPOINT", checkpoint)

    progress = tetris_main.load_launcher_training_progress()

    assert progress is not None
    assert progress.episodes == 3
    assert progress.total_steps == 500
    assert progress.best_lines == 42
    assert progress.best_reward == 123.5


def test_load_launcher_training_progress_falls_back_to_log(tmp_path, monkeypatch):
    checkpoint = tmp_path / "ai_policy.pt"
    log_path = tmp_path / "training_metrics.jsonl"
    log_path.write_text(
        "\n".join(
            [
                '{"episode": 1, "steps": 120, "reward": 10.5, "lines_cleared": 3}',
                '{"episode": 2, "steps": 140, "reward": 18.0, "lines_cleared": 5}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tetris_main, "DEFAULT_CHECKPOINT", checkpoint)

    progress = tetris_main.load_launcher_training_progress()

    assert progress is not None
    assert progress.source == "log"
    assert progress.episodes == 2
    assert progress.total_steps == 260
    assert progress.best_lines == 5
    assert progress.best_reward == 18.0


def test_stop_launch_job_requests_stop(monkeypatch):
    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self._done = threading.Event()
            self.signals = []
            self.terminated = False
            self.killed = False
            self.returncode = None

        def wait(self, timeout=None):
            if not self._done.wait(timeout):
                raise TimeoutError("process did not exit")
            self.returncode = 0
            return self.returncode

        def send_signal(self, sig):
            self.signals.append(sig)
            self._done.set()

        def terminate(self):
            self.terminated = True
            self._done.set()

        def kill(self):
            self.killed = True
            self._done.set()

        def poll(self):
            return self.returncode

    fake_process = FakeProcess()
    monkeypatch.setattr(tetris_main.subprocess, "Popen", lambda *args, **kwargs: fake_process)

    job = start_launch_job("train", ["python", "train_ai.py"])
    assert job.process is fake_process
    assert job.completed is False

    stopped = stop_launch_job(job)
    assert stopped is True
    job.thread.join(timeout=1.0)
    assert job.stop_requested is True
    assert job.completed is True
    assert job.returncode == 0
    if getattr(signal, "CTRL_BREAK_EVENT", None) is not None:
        assert fake_process.signals or fake_process.terminated


def test_stop_all_launch_jobs_stops_every_running_job(monkeypatch):
    stopped = []

    def fake_stop_launch_job(job, *, timeout=5.0):
        stopped.append(job.label)
        job.stop_requested = True
        return True

    monkeypatch.setattr(tetris_main, "stop_launch_job", fake_stop_launch_job)
    jobs = [
        LaunchJob(label="train", command=["python", "train_ai.py"]),
        LaunchJob(label="benchmark", command=["python", "benchmark_ai.py"]),
    ]

    stop_all_launch_jobs(jobs)

    assert stopped == ["train", "benchmark"]


def test_renderer_draw_start_screen_accepts_launcher_state():
    pygame.init()
    try:
        renderer = Renderer()
        surface = pygame.Surface((renderer.surface_width, renderer.surface_height))
        renderer.draw_start_screen(
            surface,
            NORMAL,
            selected_action=START_ACTIONS[0],
            status_message="Testing launcher",
            checkpoint_ready=False,
            job_active=True,
        )
    finally:
        pygame.quit()
