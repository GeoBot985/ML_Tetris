from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from tetris.api import build_snapshot
from tetris.board import Board
from tetris.difficulty import EASY, HARD, NORMAL, Difficulty
from tetris.game_state import GameState
from tetris.commands import apply_command
from tetris.piece_source import classic_uniform_source
from tetris.pieces import Piece, make_piece

from .environment import TetrisEnvironment, snapshot_to_observation
from .human_hints import load_human_hints
from .policy import API_ACTIONS, PPOAgentModel, PPOPolicy, mask_non_play_logits
from .rewards import RewardBreakdown, calculate_reward


TRAIN_ACTIONS = ("left", "right", "soft_drop", "hard_drop", "rotate_cw", "rotate_ccw", "hold")
SEARCH_ACTIONS = TRAIN_ACTIONS
ACTION_PRIORITY = {
    "hard_drop": 6,
    "soft_drop": 5,
    "rotate_cw": 4,
    "rotate_ccw": 4,
    "left": 3,
    "right": 3,
    "hold": 2,
    "pause": 0,
    "restart": 0,
}


@dataclass
class TrainingConfig:
    episodes: int = 200
    max_steps: int = 500
    rollout_steps: int = 128
    ppo_epochs: int = 4
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    gamma: float = 0.99
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    batch_size: int = 64
    supervised_epochs: int = 8
    seed: int | None = 7
    piece_source: str = "classic_uniform"
    difficulty: Difficulty = NORMAL
    checkpoint_path: Path = Path("artifacts/ai_policy.pt")
    log_path: Path = Path("artifacts/training_metrics.jsonl")
    feedback_path: Path = Path("artifacts/training_feedback.md")
    evaluation_interval: int = 10
    evaluation_episodes: int = 3
    improvement_lines_target: int = 50
    parallel_envs: int = 1
    use_shared_memory: bool = False
    verify_quantized_checkpoint: bool = True
    human_hints_path: Path | None = Path("artifacts/human_hints.jsonl")
    human_hint_weight: float = 0.05
    human_hint_decay: float = 0.995
    human_hint_batch_size: int = 32


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    steps: int
    reward: float
    score: int
    lines_cleared: int
    stack_height: int
    hole_count: int
    best_eval_reward: float
    last_action: str
    game_over: bool


@dataclass
class TrainingSummary:
    episodes: list[EpisodeMetrics] = field(default_factory=list)
    best_lines: int = 0
    best_reward: float = float("-inf")
    checkpoint_path: Path | None = None
    feedback: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_lines": self.best_lines,
            "best_reward": self.best_reward,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "episodes": [metric.__dict__ for metric in self.episodes],
            "feedback": list(self.feedback),
        }


@dataclass(frozen=True)
class TrainingProgress:
    episodes: int
    total_steps: int
    best_lines: int
    best_reward: float
    checkpoint_path: Path
    source: str = "checkpoint"

    @property
    def label(self) -> str:
        return f"{self.episodes} episodes | {self.total_steps} steps"


def _infinite_piece_source(prefix: Iterable[str] = (), fallback: Iterable[str] | None = None):
    fallback = tuple(fallback or ("I", "O", "T", "S", "Z", "J", "L"))
    while True:
        yielded = False
        for name in prefix:
            yielded = True
            yield name
        if yielded:
            prefix = ()
        for name in fallback:
            yield name


def _blank_board(width: int, height: int) -> Board:
    board = Board(width, height)
    return board


def snapshot_to_game_state(snapshot: dict[str, Any], difficulty: Difficulty = NORMAL) -> GameState:
    next_queue = list(snapshot.get("next_queue") or [])
    hold_piece_name = snapshot.get("hold_piece")
    active_piece_data = snapshot.get("active_piece") or {}
    board_data = snapshot.get("locked_board") or snapshot.get("board") or []
    queue_size = max(5, len(next_queue))
    source_prefix = [*next_queue, *("I", "O", "T", "S", "Z", "J", "L")]
    game_state = GameState(piece_source=_infinite_piece_source(source_prefix), queue_size=queue_size, difficulty=difficulty)

    board_width = len(board_data[0]) if board_data else 10
    board_height = len(board_data)
    game_state.board = _blank_board(board_width, board_height)
    for y, row in enumerate(board_data):
        for x, cell in enumerate(row):
            game_state.board.grid[y][x] = cell

    game_state.score = int(snapshot.get("score", 0))
    game_state.level = int(snapshot.get("level", 1))
    game_state.lines_cleared = int(snapshot.get("lines_cleared", 0))
    game_state.paused = bool(snapshot.get("paused", False))
    game_state.game_over = bool(snapshot.get("game_over", False))
    game_state.hold_piece = make_piece(hold_piece_name) if hold_piece_name else None
    game_state.hold_used = bool(snapshot.get("hold_used", False))
    game_state.next_queue = deque(make_piece(name) for name in next_queue)
    game_state.active_piece = make_piece(active_piece_data.get("name", "O"), int(active_piece_data.get("rotation", 0)))
    game_state.active_x = int(active_piece_data.get("x", 0))
    game_state.active_y = int(active_piece_data.get("y", 0))
    return game_state


def _snapshot_after_action(snapshot: dict[str, Any], action: str, difficulty: Difficulty) -> dict[str, Any]:
    game_state = snapshot_to_game_state(snapshot, difficulty=difficulty)
    if action != "noop":
        if action in {"left", "right", "soft_drop", "hard_drop", "rotate_cw", "rotate_ccw", "pause", "restart"}:
            apply_command(game_state, action)
    if action not in {"hard_drop", "restart"} and not game_state.game_over:
        game_state.gravity_tick()
    app_state = "game_over" if game_state.game_over else "playing"
    return build_snapshot(app_state, game_state, difficulty, [EASY, NORMAL, HARD])


def _candidate_actions(snapshot: dict[str, Any]) -> list[str]:
    if snapshot.get("game_over"):
        return ["restart"]
    if snapshot.get("paused"):
        return ["pause"]
    return list(SEARCH_ACTIONS)


def _board_from_snapshot(snapshot: dict[str, Any]) -> Board:
    rows = snapshot.get("locked_board") or snapshot.get("board") or []
    width = len(rows[0]) if rows else 10
    height = len(rows)
    board = Board(width, height)
    board.grid = [list(row) for row in rows]
    return board


def _placement_snapshot(
    snapshot: dict[str, Any],
    piece: Piece,
    x: int,
    y: int,
    cleared: int,
    board: Board,
) -> dict[str, Any]:
    return {
        "locked_board": [row[:] for row in board.grid],
        "lines_cleared": int(snapshot.get("lines_cleared", 0)) + cleared,
        "paused": False,
        "game_over": False,
        "score": int(snapshot.get("score", 0)),
        "level": int(snapshot.get("level", 1)),
        "gravity_ms": int(snapshot.get("gravity_ms", 0)),
        "active_piece": {
            "name": piece.name,
            "rotation": piece.rotation,
            "x": x,
            "y": y,
            "cells": list(piece.cells()),
        },
    }


def _best_placement(snapshot: dict[str, Any], difficulty: Difficulty = NORMAL) -> tuple[int, int, int, RewardBreakdown]:
    options = _placement_options(snapshot, difficulty)
    if not options:
        return 0, 0, 0, calculate_reward(snapshot, snapshot)
    best_option = max(options, key=lambda option: option["score"])
    return (
        best_option["rotation"],
        best_option["x"],
        best_option["y"],
        best_option["breakdown"],
    )


def _placement_options(snapshot: dict[str, Any], difficulty: Difficulty = NORMAL) -> list[dict[str, Any]]:
    board = _board_from_snapshot(snapshot)
    active_piece_data = snapshot.get("active_piece") or {}
    piece_name = active_piece_data.get("name")
    if piece_name is None:
        return []

    options: list[dict[str, Any]] = []
    width = board.width
    for rotation in range(4):
        piece = make_piece(piece_name, rotation)
        for x in range(-4, width + 4):
            if not board.can_place(piece.cells(), x, 0):
                continue
            y = 0
            while board.can_place(piece.cells(), x, y + 1):
                y += 1
            trial_board = Board(board.width, board.height)
            trial_board.grid = [row[:] for row in board.grid]
            cleared = trial_board.lock_piece(piece.cells(), x, y, piece.name)
            placement_snapshot = _placement_snapshot(snapshot, piece, x, y, cleared, trial_board)
            breakdown = calculate_reward(snapshot, placement_snapshot)
            score = breakdown.total + cleared * 0.5
            options.append(
                {
                    "rotation": rotation,
                    "x": x,
                    "y": y,
                    "breakdown": breakdown,
                    "score": score,
                    "snapshot": placement_snapshot,
                }
            )
    return options


def coach_action(snapshot: dict[str, Any], difficulty: Difficulty = NORMAL, depth: int = 2) -> tuple[str, float, RewardBreakdown]:
    if snapshot.get("game_over"):
        return "restart", 0.0, calculate_reward(snapshot, snapshot)
    if snapshot.get("paused"):
        return "pause", 0.0, calculate_reward(snapshot, snapshot)

    active_piece_data = snapshot.get("active_piece") or {}
    current_rotation = int(active_piece_data.get("rotation", 0)) % 4
    current_x = int(active_piece_data.get("x", 0))
    options = _placement_options(snapshot, difficulty=difficulty)
    if not options:
        return "hard_drop", 0.0, calculate_reward(snapshot, snapshot)

    next_queue = snapshot.get("next_queue") or []
    next_piece_name = next_queue[0] if next_queue else None
    best_option = None
    best_score = float("-inf")
    best_breakdown = calculate_reward(snapshot, snapshot)
    for option in options:
        score = option["score"]
        if next_piece_name:
            next_snapshot = {
                "locked_board": option["snapshot"]["locked_board"],
                "lines_cleared": option["snapshot"]["lines_cleared"],
                "paused": False,
                "game_over": False,
                "active_piece": {
                    "name": next_piece_name,
                    "rotation": 0,
                    "x": 0,
                    "y": 0,
                    "cells": list(make_piece(next_piece_name).cells()),
                },
            }
            _, _, _, next_breakdown = _best_placement(next_snapshot, difficulty=difficulty)
            score += 0.75 * next_breakdown.total
        if score > best_score:
            best_score = score
            best_option = option
            best_breakdown = option["breakdown"]

    assert best_option is not None

    hold_snapshot = _snapshot_after_action(snapshot, "hold", difficulty)
    hold_options = _placement_options(hold_snapshot, difficulty=difficulty)
    if hold_options:
        hold_next_queue = hold_snapshot.get("next_queue") or []
        hold_next_piece_name = hold_next_queue[0] if hold_next_queue else None
        hold_score = float("-inf")
        for option in hold_options:
            score = option["score"]
            if hold_next_piece_name:
                next_snapshot = {
                    "locked_board": option["snapshot"]["locked_board"],
                    "lines_cleared": option["snapshot"]["lines_cleared"],
                    "paused": False,
                    "game_over": False,
                    "hold_used": bool(hold_snapshot.get("hold_used", False)),
                    "active_piece": {
                        "name": hold_next_piece_name,
                        "rotation": 0,
                        "x": 0,
                        "y": 0,
                        "cells": list(make_piece(hold_next_piece_name).cells()),
                    },
                }
                _, _, _, next_breakdown = _best_placement(next_snapshot, difficulty=difficulty)
                score += 0.75 * next_breakdown.total
            hold_score = max(hold_score, score)
        if hold_score > best_score:
            return "hold", hold_score, calculate_reward(snapshot, hold_snapshot)

    target_rotation = best_option["rotation"]
    target_x = best_option["x"]

    if current_rotation != target_rotation:
        clockwise_steps = (target_rotation - current_rotation) % 4
        if clockwise_steps in {1, 2}:
            return "rotate_cw", best_breakdown.total, best_breakdown
        return "rotate_ccw", best_breakdown.total, best_breakdown

    if current_x < target_x:
        return "right", best_breakdown.total, best_breakdown
    if current_x > target_x:
        return "left", best_breakdown.total, best_breakdown
    return "hard_drop", best_breakdown.total, best_breakdown


def action_index(action: str) -> int:
    if action not in API_ACTIONS:
        raise ValueError(f"Unsupported policy action: {action}")
    return API_ACTIONS.index(action)


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _suggest_tuning(summary: TrainingSummary) -> list[str]:
    if not summary.episodes:
        return ["No episodes were recorded."]
    recent = summary.episodes[-10:]
    avg_lines = sum(item.lines_cleared for item in recent) / len(recent)
    avg_holes = sum(item.hole_count for item in recent) / len(recent)
    avg_stack = sum(item.stack_height for item in recent) / len(recent)
    suggestions = []
    if avg_lines < 1:
        suggestions.append("Increase exploration pressure or reduce entropy decay; the agent is not finding clears.")
    if avg_holes > 5:
        suggestions.append("Increase hole penalties or add a stronger anti-gap reward term.")
    if avg_stack > 12:
        suggestions.append("Increase stack-height penalties or make hard-drop placements more attractive.")
    if not suggestions:
        suggestions.append("Current reward shaping is stable; focus on longer rollouts or more training episodes.")
    return suggestions


def _piece_source_factory(config: TrainingConfig, seed_offset: int = 0):
    if config.piece_source == "classic_uniform":
        seed = None if config.seed is None else config.seed + seed_offset
        return lambda: classic_uniform_source(seed=seed)
    if config.piece_source == "seven_bag":
        return None
    raise ValueError(f"Unsupported piece source: {config.piece_source}")


def _train_policy_parallel(config: TrainingConfig) -> TrainingSummary:
    from .environment import build_observation_layout, observation_to_snapshot
    from .vectorized import make_vec_env

    torch.manual_seed(config.seed or 0)
    random.seed(config.seed or 0)

    vec_env = make_vec_env(
        num_envs=config.parallel_envs,
        difficulty=config.difficulty,
        seed=config.seed,
        use_shared_memory=config.use_shared_memory,
        piece_source=config.piece_source,
    )
    try:
        observations = vec_env.reset()
        policy = PPOPolicy.from_observation_dim(observations.shape[1])
        model = policy.model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        summary = TrainingSummary()
        layout = build_observation_layout()

        episode_buffers = [
            {
                "observations": [],
                "action_targets": [],
                "value_targets": [],
                "weights": [],
                "reward": 0.0,
                "steps": 0,
                "best_eval_reward": float("-inf"),
                "last_action": "noop",
            }
            for _ in range(config.parallel_envs)
        ]
        completed_episodes = 0
        interrupted = False

        try:
            while completed_episodes < config.episodes:
                snapshot_batch = [observation_to_snapshot(observations[index], layout=layout) for index in range(config.parallel_envs)]
                actions: list[int] = []
                for index, snapshot in enumerate(snapshot_batch):
                    coach, coach_reward, reward_breakdown = coach_action(snapshot, difficulty=config.difficulty)
                    buffer = episode_buffers[index]
                    buffer["observations"].append(torch.as_tensor(observations[index].copy(), dtype=torch.float32))
                    buffer["action_targets"].append(action_index(coach))
                    buffer["value_targets"].append(float(coach_reward))
                    line_clear_signal = reward_breakdown.line_clear_reward + reward_breakdown.line_clear_complexity_reward
                    buffer["weights"].append(1.0 + max(0.0, line_clear_signal))
                    buffer["steps"] += 1
                    buffer["last_action"] = coach
                    actions.append(action_index(coach))

                next_observations, rewards, dones, infos = vec_env.step(actions)

                for index in range(config.parallel_envs):
                    buffer = episode_buffers[index]
                    buffer["reward"] += float(rewards[index])
                    buffer["best_eval_reward"] = max(buffer["best_eval_reward"], float(rewards[index]))
                    if dones[index]:
                        info = infos[index]
                        breakdown = info.get("reward_breakdown")
                        if breakdown is None:
                            breakdown = calculate_reward(snapshot_batch[index], snapshot_batch[index])
                        profile = breakdown.profile
                        metrics = EpisodeMetrics(
                            episode=completed_episodes + 1,
                            steps=int(buffer["steps"]),
                            reward=float(buffer["reward"]),
                            score=int(info.get("score", 0)),
                            lines_cleared=int(info.get("lines_cleared", 0)),
                            stack_height=int(profile.stack_height),
                            hole_count=int(profile.hole_count),
                            best_eval_reward=float(buffer["best_eval_reward"]),
                            last_action=buffer["last_action"],
                            game_over=bool(info.get("game_over", False)),
                        )
                        _train_batch(
                            model,
                            optimizer,
                            buffer["observations"],
                            buffer["action_targets"],
                            buffer["value_targets"],
                            buffer["weights"],
                            config,
                        )
                        summary.episodes.append(metrics)
                        summary.best_lines = max(summary.best_lines, metrics.lines_cleared)
                        summary.best_reward = max(summary.best_reward, metrics.reward)
                        _write_jsonl(
                            config.log_path,
                            {
                                "episode": metrics.episode,
                                "steps": metrics.steps,
                                "reward": metrics.reward,
                                "score": metrics.score,
                                "lines_cleared": metrics.lines_cleared,
                                "stack_height": metrics.stack_height,
                                "hole_count": metrics.hole_count,
                                "best_eval_reward": metrics.best_eval_reward,
                                "last_action": metrics.last_action,
                                "game_over": metrics.game_over,
                            },
                        )
                        buffer["observations"].clear()
                        buffer["action_targets"].clear()
                        buffer["value_targets"].clear()
                        buffer["weights"].clear()
                        buffer["reward"] = 0.0
                        buffer["steps"] = 0
                        buffer["best_eval_reward"] = float("-inf")
                        buffer["last_action"] = "noop"
                        completed_episodes += 1
                        if completed_episodes % config.evaluation_interval == 0:
                            eval_metrics = evaluate_policy(policy, config, episodes=config.evaluation_episodes)
                            summary.feedback.extend(_suggest_tuning(summary))
                            if eval_metrics["best_lines"] >= config.improvement_lines_target:
                                save_checkpoint(config.checkpoint_path, model, config, summary)
                        if completed_episodes >= config.episodes:
                            break

                observations = next_observations
        except KeyboardInterrupt:
            interrupted = True
            summary.feedback.append("Training interrupted; saving partial checkpoint.")

        save_checkpoint(config.checkpoint_path, model, config, summary)
        summary.checkpoint_path = config.checkpoint_path
        summary.feedback.extend(_suggest_tuning(summary))
        if config.verify_quantized_checkpoint and not interrupted:
            load_policy_from_checkpoint(config.checkpoint_path, quantize=True)
        write_feedback_report(config.feedback_path, summary)
        return summary
    finally:
        vec_env.close()


def _train_policy_bootstrap(config: TrainingConfig) -> TrainingSummary:
    if config.parallel_envs > 1:
        return _train_policy_parallel(config)

    torch.manual_seed(config.seed or 0)
    random.seed(config.seed or 0)

    env = TetrisEnvironment(difficulty=config.difficulty, piece_source_factory=_piece_source_factory(config))
    policy = PPOPolicy.from_snapshot(env.snapshot())
    model = policy.model
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    summary = TrainingSummary()

    running_reward = 0.0
    running_steps = 0
    interrupted = False
    try:
        for episode in range(1, config.episodes + 1):
            env.piece_source_factory = _piece_source_factory(config, seed_offset=episode)
            observation, info = env.reset(seed=None if config.seed is None else config.seed + episode)
            episode_reward = 0.0
            best_eval_reward = float("-inf")
            last_action = "noop"

            observations: list[torch.Tensor] = []
            action_targets: list[int] = []
            value_targets: list[float] = []
            weights: list[float] = []

            for step in range(config.max_steps):
                snapshot = env.snapshot()
                coach, coach_reward, reward_breakdown = coach_action(snapshot, difficulty=config.difficulty)
                next_observation, reward, terminated, truncated, step_info = env.step(coach)

                observations.append(torch.as_tensor(observation, dtype=torch.float32))
                action_targets.append(action_index(coach))
                value_targets.append(float(coach_reward))
                line_clear_signal = reward_breakdown.line_clear_reward + reward_breakdown.line_clear_complexity_reward
                weights.append(1.0 + max(0.0, line_clear_signal))

                observation = next_observation
                episode_reward += reward
                last_action = coach
                best_eval_reward = max(best_eval_reward, reward)
                running_steps += 1

                if terminated or truncated:
                    break

            for _ in range(config.supervised_epochs):
                _train_batch(model, optimizer, observations, action_targets, value_targets, weights, config)

            final_snapshot = env.snapshot()
            profile = final_snapshot.get("reward_breakdown") if False else None
            board_breakdown = calculate_reward(final_snapshot, final_snapshot)
            metrics = EpisodeMetrics(
                episode=episode,
                steps=len(observations),
                reward=episode_reward,
                score=int(env.game_state.score),
                lines_cleared=int(env.game_state.lines_cleared),
                stack_height=board_breakdown.profile.stack_height,
                hole_count=board_breakdown.profile.hole_count,
                best_eval_reward=best_eval_reward,
                last_action=last_action,
                game_over=bool(env.game_state.game_over),
            )
            summary.episodes.append(metrics)
            summary.best_lines = max(summary.best_lines, metrics.lines_cleared)
            summary.best_reward = max(summary.best_reward, metrics.reward)
            _write_jsonl(
                config.log_path,
                {
                    "episode": metrics.episode,
                    "steps": metrics.steps,
                    "reward": metrics.reward,
                    "score": metrics.score,
                    "lines_cleared": metrics.lines_cleared,
                    "stack_height": metrics.stack_height,
                    "hole_count": metrics.hole_count,
                    "best_eval_reward": metrics.best_eval_reward,
                    "last_action": metrics.last_action,
                    "game_over": metrics.game_over,
                },
            )

            if episode % config.evaluation_interval == 0:
                eval_metrics = evaluate_policy(policy, config, episodes=config.evaluation_episodes)
                summary.feedback.extend(_suggest_tuning(summary))
                if eval_metrics["best_lines"] >= config.improvement_lines_target:
                    save_checkpoint(config.checkpoint_path, model, config, summary)

            running_reward += episode_reward
    except KeyboardInterrupt:
        interrupted = True
        summary.feedback.append("Training interrupted; saving partial checkpoint.")

    save_checkpoint(config.checkpoint_path, model, config, summary)
    summary.checkpoint_path = config.checkpoint_path
    summary.feedback.extend(_suggest_tuning(summary))
    if config.verify_quantized_checkpoint and not interrupted:
        load_policy_from_checkpoint(config.checkpoint_path, quantize=True)
    write_feedback_report(config.feedback_path, summary)
    return summary


def _train_batch(
    model: PPOAgentModel,
    optimizer: torch.optim.Optimizer,
    observations: list[torch.Tensor],
    action_targets: list[int],
    value_targets: list[float],
    weights: list[float],
    config: TrainingConfig,
) -> None:
    if not observations:
        return
    batch_obs = torch.stack(observations)
    batch_actions = torch.tensor(action_targets, dtype=torch.long)
    batch_values = torch.tensor(value_targets, dtype=torch.float32)
    batch_weights = torch.tensor(weights, dtype=torch.float32)

    model.train()
    logits, values = model(batch_obs)
    masked_logits = mask_non_play_logits(logits)
    actor_loss = F.cross_entropy(masked_logits, batch_actions, reduction="none")
    actor_loss = (actor_loss * batch_weights).mean()
    critic_loss = F.mse_loss(values, batch_values)
    entropy = Categorical(logits=masked_logits).entropy().mean()
    loss = actor_loss + config.value_weight * critic_loss - config.entropy_weight * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.eval()


def evaluate_policy(policy: PPOPolicy, config: TrainingConfig, episodes: int = 1) -> dict[str, Any]:
    env = TetrisEnvironment(difficulty=config.difficulty, piece_source_factory=_piece_source_factory(config, seed_offset=1000))
    best_lines = 0
    rewards = []
    for episode in range(episodes):
        env.piece_source_factory = _piece_source_factory(config, seed_offset=1000 + episode)
        env.reset(seed=None if config.seed is None else config.seed + 1000 + episode)
        total_reward = 0.0
        for _ in range(config.max_steps):
            decision = policy.act_from_snapshot(env.snapshot(), deterministic=True)
            _, reward, terminated, truncated, _ = env.step(decision.action)
            total_reward += reward
            if terminated or truncated:
                break
        best_lines = max(best_lines, int(env.game_state.lines_cleared))
        rewards.append(total_reward)
    return {
        "best_lines": best_lines,
        "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
    }


def save_checkpoint(path: Path, model: PPOAgentModel, config: TrainingConfig, summary: TrainingSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "observation_dim": snapshot_to_observation(
                TetrisEnvironment(difficulty=config.difficulty, piece_source_factory=_piece_source_factory(config)).snapshot()
            ).shape[0],
            "action_names": API_ACTIONS,
            "config": {
                "episodes": config.episodes,
                "max_steps": config.max_steps,
                "learning_rate": config.learning_rate,
                "gamma": config.gamma,
                "entropy_weight": config.entropy_weight,
                "value_weight": config.value_weight,
                "difficulty": config.difficulty.name,
                "piece_source": config.piece_source,
                "training_algorithm": "ppo",
            },
            "summary": summary.as_dict(),
        },
        path,
    )


def load_policy_from_checkpoint(
    path: Path,
    device: str | torch.device | None = None,
    *,
    quantize: bool = False,
) -> PPOPolicy:
    payload = torch.load(path, map_location=device or "cpu", weights_only=True)
    model = PPOAgentModel(payload["observation_dim"])
    model.load_state_dict(payload["model_state_dict"])
    policy = PPOPolicy(model, device=device, fallback_action_fn=None)
    if quantize:
        from .deployment import quantize_policy

        policy = quantize_policy(policy)
    return policy


def load_training_progress(path: Path) -> TrainingProgress | None:
    def from_checkpoint(checkpoint_path: Path) -> TrainingProgress | None:
        if not checkpoint_path.exists():
            return None
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        summary = payload.get("summary") or {}
        episodes = summary.get("episodes") or []
        total_steps = sum(int(item.get("steps", 0)) for item in episodes)
        best_reward = float(summary.get("best_reward", 0.0))
        return TrainingProgress(
            episodes=len(episodes),
            total_steps=total_steps,
            best_lines=int(summary.get("best_lines", 0)),
            best_reward=best_reward if best_reward != float("-inf") else 0.0,
            checkpoint_path=checkpoint_path,
            source="checkpoint",
        )

    def from_log(log_path: Path) -> TrainingProgress | None:
        if not log_path.exists():
            return None
        episodes = 0
        total_steps = 0
        best_lines = 0
        best_reward = float("-inf")
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "training_episode" and "episode" not in payload:
                continue
            episodes += 1
            total_steps += int(payload.get("steps", 0))
            best_lines = max(best_lines, int(payload.get("lines_cleared", 0)))
            best_reward = max(best_reward, float(payload.get("reward", 0.0)))
        if episodes == 0:
            return None
        return TrainingProgress(
            episodes=episodes,
            total_steps=total_steps,
            best_lines=best_lines,
            best_reward=best_reward if best_reward != float("-inf") else 0.0,
            checkpoint_path=log_path,
            source="log",
        )

    checkpoint_progress = from_checkpoint(path)
    log_path = path.with_name("training_metrics.jsonl")
    log_progress = from_log(log_path)

    if checkpoint_progress is None:
        return log_progress
    if log_progress is None:
        return checkpoint_progress
    if log_progress.episodes > checkpoint_progress.episodes or log_progress.total_steps > checkpoint_progress.total_steps:
        return log_progress
    return checkpoint_progress


def write_feedback_report(path: Path, summary: TrainingSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Training Feedback", ""]
    if summary.episodes:
        recent = summary.episodes[-10:]
        avg_reward = sum(item.reward for item in recent) / len(recent)
        avg_lines = sum(item.lines_cleared for item in recent) / len(recent)
        avg_holes = sum(item.hole_count for item in recent) / len(recent)
        avg_stack = sum(item.stack_height for item in recent) / len(recent)
        lines.extend(
            [
                f"- Recent average reward: {avg_reward:.2f}",
                f"- Recent average lines cleared: {avg_lines:.2f}",
                f"- Recent average hole count: {avg_holes:.2f}",
                f"- Recent average stack height: {avg_stack:.2f}",
                "",
                "## Suggested Tweaks",
            ]
        )
        for suggestion in _suggest_tuning(summary):
            lines.append(f"- {suggestion}")
    else:
        lines.append("- No training episodes were completed.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    *,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
        raise ValueError("rewards, values, and dones must be 1D tensors")
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have the same shape")

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=torch.float32, device=rewards.device)
    next_value = last_value
    for index in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * mask - values[index]
        gae = delta + gamma * lambda_ * mask * gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages + values
    return advantages, returns


def _prepare_rollout_batch(
    observations: list[torch.Tensor],
    actions: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    rewards: list[torch.Tensor],
    dones: list[torch.Tensor],
    values: list[torch.Tensor],
    last_values: torch.Tensor,
    *,
    gamma: float,
    lambda_: float,
) -> RolloutBatch:
    obs_tensor = torch.stack(observations)  # [T, N, obs_dim]
    actions_tensor = torch.stack(actions)  # [T, N]
    log_probs_tensor = torch.stack(old_log_probs)  # [T, N]
    rewards_tensor = torch.stack(rewards)  # [T, N]
    dones_tensor = torch.stack(dones)  # [T, N]
    values_tensor = torch.stack(values)  # [T, N]

    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for env_index in range(rewards_tensor.shape[1]):
        env_advantages, env_returns = compute_gae(
            rewards_tensor[:, env_index],
            values_tensor[:, env_index],
            dones_tensor[:, env_index],
            last_values[env_index],
            gamma=gamma,
            lambda_=lambda_,
        )
        advantages.append(env_advantages)
        returns.append(env_returns)

    advantages_tensor = torch.stack(advantages, dim=1)
    returns_tensor = torch.stack(returns, dim=1)

    flat_observations = obs_tensor.reshape(-1, obs_tensor.shape[-1])
    flat_actions = actions_tensor.reshape(-1)
    flat_log_probs = log_probs_tensor.reshape(-1)
    flat_returns = returns_tensor.reshape(-1)
    flat_advantages = advantages_tensor.reshape(-1)

    flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std(unbiased=False) + 1e-8)
    return RolloutBatch(
        observations=flat_observations,
        actions=flat_actions,
        old_log_probs=flat_log_probs,
        returns=flat_returns,
        advantages=flat_advantages,
    )


def _ppo_update(
    model: PPOAgentModel,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    config: TrainingConfig,
    human_hints: tuple[torch.Tensor, torch.Tensor] | None = None,
    human_hint_weight: float = 0.0,
) -> None:
    batch_size = batch.observations.shape[0]
    minibatch_size = max(1, min(config.batch_size, batch_size))
    model.train()
    for _ in range(max(1, config.ppo_epochs)):
        permutation = torch.randperm(batch_size, device=batch.observations.device)
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            minibatch_obs = batch.observations[indices]
            minibatch_actions = batch.actions[indices]
            minibatch_old_log_probs = batch.old_log_probs[indices]
            minibatch_returns = batch.returns[indices]
            minibatch_advantages = batch.advantages[indices]

            logits, values = model(minibatch_obs)
            dist = Categorical(logits=mask_non_play_logits(logits))
            new_log_probs = dist.log_prob(minibatch_actions)
            ratio = torch.exp(new_log_probs - minibatch_old_log_probs)
            unclipped = ratio * minibatch_advantages
            clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * minibatch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, minibatch_returns)
            entropy_bonus = dist.entropy().mean()
            loss = policy_loss + config.value_weight * value_loss - config.entropy_weight * entropy_bonus
            if human_hints is not None and human_hint_weight > 0:
                hint_observations, hint_actions = human_hints
                hint_size = hint_observations.shape[0]
                hint_batch_size = max(1, min(config.human_hint_batch_size, hint_size))
                hint_indices = torch.randint(0, hint_size, (hint_batch_size,))
                hint_obs = hint_observations[hint_indices].to(minibatch_obs.device)
                hint_targets = hint_actions[hint_indices].to(minibatch_actions.device)
                hint_logits, _ = model(hint_obs)
                loss = loss + human_hint_weight * F.cross_entropy(mask_non_play_logits(hint_logits), hint_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()


def _make_episode_metrics(
    episode_number: int,
    buffer: dict[str, Any],
    info: dict[str, Any],
) -> EpisodeMetrics:
    breakdown = info.get("reward_breakdown")
    profile = getattr(breakdown, "profile", None)
    return EpisodeMetrics(
        episode=episode_number,
        steps=int(buffer["steps"]),
        reward=float(buffer["reward"]),
        score=int(info.get("score", 0)),
        lines_cleared=int(info.get("lines_cleared", 0)),
        stack_height=int(getattr(profile, "stack_height", 0)),
        hole_count=int(getattr(profile, "hole_count", 0)),
        best_eval_reward=float(buffer["best_eval_reward"]),
        last_action=str(buffer["last_action"]),
        game_over=bool(info.get("game_over", False)),
    )


def _train_policy_ppo(config: TrainingConfig) -> TrainingSummary:
    from .environment import build_observation_layout, observation_to_snapshot
    from .vectorized import make_vec_env

    torch.manual_seed(config.seed or 0)
    random.seed(config.seed or 0)

    vec_env = make_vec_env(
        num_envs=config.parallel_envs,
        difficulty=config.difficulty,
        seed=config.seed,
        use_shared_memory=config.use_shared_memory,
        piece_source=config.piece_source,
    )
    try:
        observations = vec_env.reset()
        policy = PPOPolicy.from_observation_dim(observations.shape[1])
        model = policy.model
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        summary = TrainingSummary()
        layout = build_observation_layout()
        num_envs = observations.shape[0]
        rollout_steps = max(1, min(config.rollout_steps, config.max_steps))
        human_hints = None
        if config.human_hints_path is not None and config.human_hint_weight > 0:
            human_hints = load_human_hints(config.human_hints_path, observation_dim=observations.shape[1])
            if human_hints is not None:
                summary.feedback.append(f"Loaded {human_hints[0].shape[0]} human hint samples.")

        episode_buffers = [
            {
                "reward": 0.0,
                "steps": 0,
                "best_eval_reward": float("-inf"),
                "last_action": "noop",
            }
            for _ in range(num_envs)
        ]
        completed_episodes = 0
        interrupted = False
        update_count = 0

        try:
            while completed_episodes < config.episodes:
                rollout_observations: list[torch.Tensor] = []
                rollout_actions: list[torch.Tensor] = []
                rollout_log_probs: list[torch.Tensor] = []
                rollout_rewards: list[torch.Tensor] = []
                rollout_dones: list[torch.Tensor] = []
                rollout_values: list[torch.Tensor] = []

                for _ in range(rollout_steps):
                    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.actor_head.weight.device)
                    with torch.no_grad():
                        dist, values = model.distribution(obs_tensor)
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)

                    next_observations, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

                    rollout_observations.append(obs_tensor.detach().cpu())
                    rollout_actions.append(actions.detach().cpu())
                    rollout_log_probs.append(log_probs.detach().cpu())
                    rollout_rewards.append(torch.as_tensor(rewards, dtype=torch.float32))
                    rollout_dones.append(torch.as_tensor(dones, dtype=torch.float32))
                    rollout_values.append(values.detach().cpu())

                    for env_index in range(num_envs):
                        buffer = episode_buffers[env_index]
                        buffer["reward"] += float(rewards[env_index])
                        buffer["steps"] += 1
                        buffer["best_eval_reward"] = max(buffer["best_eval_reward"], float(rewards[env_index]))
                        buffer["last_action"] = API_ACTIONS[int(actions[env_index].item())]
                        if dones[env_index]:
                            episode_metrics = _make_episode_metrics(
                                completed_episodes + 1,
                                buffer,
                                infos[env_index],
                            )
                            summary.episodes.append(episode_metrics)
                            summary.best_lines = max(summary.best_lines, episode_metrics.lines_cleared)
                            summary.best_reward = max(summary.best_reward, episode_metrics.reward)
                            _write_jsonl(
                                config.log_path,
                                {
                                    "type": "training_episode",
                                    "algorithm": "ppo",
                                    "episode": episode_metrics.episode,
                                    "steps": episode_metrics.steps,
                                    "reward": episode_metrics.reward,
                                    "score": episode_metrics.score,
                                    "lines_cleared": episode_metrics.lines_cleared,
                                    "stack_height": episode_metrics.stack_height,
                                    "hole_count": episode_metrics.hole_count,
                                    "best_eval_reward": episode_metrics.best_eval_reward,
                                    "last_action": episode_metrics.last_action,
                                    "game_over": episode_metrics.game_over,
                                },
                            )
                            buffer["reward"] = 0.0
                            buffer["steps"] = 0
                            buffer["best_eval_reward"] = float("-inf")
                            buffer["last_action"] = "noop"
                            completed_episodes += 1
                            if completed_episodes >= config.episodes:
                                break

                    observations = next_observations
                    if completed_episodes >= config.episodes:
                        break

                with torch.no_grad():
                    final_obs = torch.as_tensor(observations, dtype=torch.float32, device=model.actor_head.weight.device)
                    _, last_values = model.distribution(final_obs)

                rollout_batch = _prepare_rollout_batch(
                    rollout_observations,
                    rollout_actions,
                    rollout_log_probs,
                    rollout_rewards,
                    rollout_dones,
                    rollout_values,
                    last_values.detach().cpu(),
                    gamma=config.gamma,
                    lambda_=config.gae_lambda,
                )
                hint_weight = config.human_hint_weight * (config.human_hint_decay**update_count)
                _ppo_update(model, optimizer, rollout_batch, config, human_hints=human_hints, human_hint_weight=hint_weight)
                update_count += 1

                if completed_episodes and completed_episodes % config.evaluation_interval == 0:
                    eval_metrics = evaluate_policy(policy, config, episodes=config.evaluation_episodes)
                    summary.feedback.extend(_suggest_tuning(summary))
                    if eval_metrics["best_lines"] >= config.improvement_lines_target:
                        save_checkpoint(config.checkpoint_path, model, config, summary)
        except KeyboardInterrupt:
            interrupted = True
            summary.feedback.append("Training interrupted; saving partial checkpoint.")

        save_checkpoint(config.checkpoint_path, model, config, summary)
        summary.checkpoint_path = config.checkpoint_path
        summary.feedback.extend(_suggest_tuning(summary))
        if config.verify_quantized_checkpoint and not interrupted:
            load_policy_from_checkpoint(config.checkpoint_path, quantize=True)
        write_feedback_report(config.feedback_path, summary)
        return summary
    finally:
        vec_env.close()


def train_policy(config: TrainingConfig) -> TrainingSummary:
    return _train_policy_ppo(config)
