"""Placement-level Tetris RL: the approach that actually lets a learned net play well.

Why this module exists
----------------------
The original micro-action setup (emit left/right/rotate/drop one step at a time)
is the worst case for RL on Tetris: long horizon, sparse reward, brutal credit
assignment. Empirically a flat MLP trained that way plays at ~1 line.

This module reframes the problem the standard way:

* **Action space = placement slots.** The agent picks *where the current piece
  lands* — a (rotation, column) pair, 4 x board_width slots, illegal ones masked.
  Each action places one piece, so reward is immediate and episodes are short.
* **CNN encoder.** Optimal placement is spatial, so the policy convolves over the
  board grid instead of flattening it.
* **DAgger warm-start.** Behaviour cloning on expert-only boards fails to
  generalise (the coach only ever visits clean boards). DAgger rolls out the
  *learner*, relabels the boards it actually reaches with the coach, and
  aggregates — covering the learner's own mistake distribution.
* **PPO fine-tune.** Once warm-started onto a sensible manifold, PPO polishes the
  policy with the shaped reward.

Everything here is self-contained and reuses the existing game logic, board
profiling, and reward shaping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from tetris.api import build_snapshot
from tetris.board import Board
from tetris.difficulty import EASY, HARD, NORMAL, Difficulty
from tetris.game_state import GameState
from tetris.piece_source import seven_bag_piece_source
from tetris.pieces import all_piece_names, make_piece

from .rewards import calculate_reward

PIECE_NAMES = tuple(all_piece_names())
BOARD_W = 10
BOARD_H = 20
N_ROTATIONS = 4
N_SLOTS = N_ROTATIONS * BOARD_W  # 40 placement actions
AUX_DIM = len(PIECE_NAMES) * 2   # active one-hot + next one-hot


# ──────────────────────────────────────────────────────────── observation

def board_planes(game_state: GameState) -> np.ndarray:
    """(1, H, W) float32 occupancy grid of locked cells."""
    grid = game_state.board.grid
    planes = np.zeros((1, BOARD_H, BOARD_W), dtype=np.float32)
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell is not None:
                planes[0, y, x] = 1.0
    return planes


def aux_features(game_state: GameState) -> np.ndarray:
    """Active-piece and next-piece identity as a flat one-hot vector."""
    vec = np.zeros(AUX_DIM, dtype=np.float32)
    if game_state.active_piece is not None:
        vec[PIECE_NAMES.index(game_state.active_piece.name)] = 1.0
    if game_state.next_queue:
        nxt = game_state.next_queue[0].name
        vec[len(PIECE_NAMES) + PIECE_NAMES.index(nxt)] = 1.0
    return vec


def observe(game_state: GameState) -> tuple[np.ndarray, np.ndarray]:
    return board_planes(game_state), aux_features(game_state)


# ──────────────────────────────────────────────────────────── placements

@dataclass(frozen=True)
class Placement:
    slot: int
    rotation: int
    x: int
    y: int
    piece: Any


def legal_placements(game_state: GameState) -> dict[int, Placement]:
    """Map slot index -> Placement for every reachable hard-drop of the active piece."""
    out: dict[int, Placement] = {}
    piece = game_state.active_piece
    if piece is None or game_state.game_over:
        return out
    board = game_state.board
    for rotation in range(N_ROTATIONS):
        rotated = make_piece(piece.name, rotation)
        cells = rotated.cells()
        for x in range(BOARD_W):
            if not board.can_place(cells, x, 0):
                continue
            y = 0
            while board.can_place(cells, x, y + 1):
                y += 1
            out[rotation * BOARD_W + x] = Placement(rotation * BOARD_W + x, rotation, x, y, rotated)
    return out


def legal_mask(game_state: GameState) -> np.ndarray:
    mask = np.zeros(N_SLOTS, dtype=bool)
    for slot in legal_placements(game_state):
        mask[slot] = True
    return mask


def apply_placement(game_state: GameState, placement: Placement) -> None:
    game_state.active_piece = placement.piece
    game_state.active_x = placement.x
    game_state.active_y = placement.y
    game_state.lock_active_piece()


# ──────────────────────────────────────────────────────────── coach expert

def _snapshot(game_state: GameState, difficulty: Difficulty) -> dict[str, Any]:
    app = "game_over" if game_state.game_over else "playing"
    return build_snapshot(app, game_state, difficulty, [EASY, NORMAL, HARD])


LOOKAHEAD_WEIGHT = 0.7


def _best_placement_score(board: Board, piece_name: str, prev_snapshot: dict[str, Any]) -> float:
    """Best one-step shaped reward achievable for ``piece_name`` on ``board``.

    Used for one-piece lookahead. If the piece cannot be placed at all (the board is
    topped out), returns a large negative score so such lines of play are avoided.
    """
    base_lines = int(prev_snapshot.get("lines_cleared", 0))
    best = float("-inf")
    for rotation in range(N_ROTATIONS):
        cells = make_piece(piece_name, rotation).cells()
        for x in range(BOARD_W):
            if not board.can_place(cells, x, 0):
                continue
            y = 0
            while board.can_place(cells, x, y + 1):
                y += 1
            trial = Board(board.width, board.height)
            trial.grid = [row[:] for row in board.grid]
            cleared = trial.lock_piece(cells, x, y, piece_name)
            placed = {
                "locked_board": [row[:] for row in trial.grid],
                "lines_cleared": base_lines + cleared,
                "paused": False,
                "game_over": False,
            }
            score = calculate_reward(prev_snapshot, placed).total + cleared * 0.5
            best = max(best, score)
    return best if best != float("-inf") else -10.0


def coach_slot(game_state: GameState, difficulty: Difficulty = NORMAL, lookahead: bool = True) -> int | None:
    """Expert placement: the slot with the best shaped reward, with one-piece lookahead.

    Evaluates every legal placement against ``calculate_reward`` (holes, bumpiness,
    wells, height, line clears) and, when ``lookahead`` is set and a next piece is
    known, adds the discounted best response for that next piece. The lookahead is
    what lifts the coach from ~8 to ~15+ lines, so it is on by default. (DAgger data
    collection may pass ``lookahead=False`` to trade quality for speed.)
    """
    placements = legal_placements(game_state)
    if not placements:
        return None
    current = _snapshot(game_state, difficulty)
    base_lines = int(current.get("lines_cleared", 0))
    next_piece = game_state.next_queue[0].name if game_state.next_queue else None
    best_slot, best_score = None, float("-inf")
    for slot, p in placements.items():
        trial = Board(game_state.board.width, game_state.board.height)
        trial.grid = [row[:] for row in game_state.board.grid]
        cleared = trial.lock_piece(p.piece.cells(), p.x, p.y, p.piece.name)
        placed = {
            "locked_board": [row[:] for row in trial.grid],
            "lines_cleared": base_lines + cleared,
            "paused": False,
            "game_over": False,
        }
        score = calculate_reward(current, placed).total + cleared * 0.5
        if lookahead and next_piece is not None:
            score += LOOKAHEAD_WEIGHT * _best_placement_score(trial, next_piece, placed)
        if score > best_score:
            best_score, best_slot = score, slot
    return best_slot


# ──────────────────────────────────────────────────────────── environment

@dataclass
class PlacementStep:
    observation: tuple[np.ndarray, np.ndarray]
    reward: float
    terminated: bool
    info: dict[str, Any]


class PlacementEnv:
    """Gym-like environment where one action places one piece."""

    n_slots = N_SLOTS

    def __init__(self, difficulty: Difficulty = NORMAL, queue_size: int = 5, max_pieces: int = 300):
        self.difficulty = difficulty
        self.queue_size = queue_size
        self.max_pieces = max_pieces
        self._seed: int | None = None
        self.game_state = self._new_game()
        self._pieces = 0

    def _new_game(self) -> GameState:
        rng = random.Random(self._seed)
        return GameState(
            piece_source=seven_bag_piece_source(shuffle=rng.shuffle),
            queue_size=self.queue_size,
            difficulty=self.difficulty,
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self._seed = seed
        self.game_state = self._new_game()
        self._pieces = 0
        return observe(self.game_state)

    def legal_mask(self) -> np.ndarray:
        return legal_mask(self.game_state)

    def coach_slot(self) -> int | None:
        return coach_slot(self.game_state, self.difficulty)

    def step(self, slot: int) -> PlacementStep:
        gs = self.game_state
        prev = _snapshot(gs, self.difficulty)
        placements = legal_placements(gs)
        placement = placements.get(int(slot))
        if placement is None:
            # Illegal slot chosen: penalise and end. (Masking should prevent this.)
            return PlacementStep(observe(gs), -5.0, True, {"illegal": True, "lines": gs.lines_cleared})
        apply_placement(gs, placement)
        self._pieces += 1
        new = _snapshot(gs, self.difficulty)
        reward = calculate_reward(prev, new).total
        terminated = gs.game_over or self._pieces >= self.max_pieces
        info = {"lines": gs.lines_cleared, "score": gs.score, "game_over": gs.game_over}
        return PlacementStep(observe(gs), reward, terminated, info)


# ──────────────────────────────────────────────────────────── CNN policy

class CNNActorCritic(nn.Module):
    def __init__(self, channels: int = 32, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out = channels * BOARD_H * BOARD_W
        # Dropout regularises the large conv->dense projection. Empirically it lifts
        # held-out coach-imitation accuracy from ~47% to ~57% (the policy overfits
        # badly without it), which is the difference between dying early and playing on.
        self.trunk = nn.Sequential(nn.Linear(conv_out + AUX_DIM, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.actor = nn.Linear(hidden, N_SLOTS)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, planes: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if planes.dim() == 3:
            planes = planes.unsqueeze(0)
        if aux.dim() == 1:
            aux = aux.unsqueeze(0)
        h = self.conv(planes.float()).flatten(1)
        h = self.trunk(torch.cat([h, aux.float()], dim=1))
        return self.actor(h), self.critic(h).squeeze(-1)


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.where(mask, logits, torch.full_like(logits, neg_inf))


class PlacementPolicy:
    """Inference wrapper around CNNActorCritic with action masking."""

    def __init__(self, model: CNNActorCritic, device: str | torch.device | None = None):
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.model.eval()

    def act(self, game_state: GameState, deterministic: bool = True) -> int | None:
        mask_np = legal_mask(game_state)
        if not mask_np.any():
            return None
        planes, aux = observe(game_state)
        with torch.no_grad():
            logits, _ = self.model(
                torch.as_tensor(planes, device=self.device),
                torch.as_tensor(aux, device=self.device),
            )
            mask = torch.as_tensor(mask_np, device=self.device).unsqueeze(0)
            masked = _masked_logits(logits, mask)
            if deterministic:
                return int(torch.argmax(masked, dim=-1).item())
            return int(Categorical(logits=masked).sample().item())


# ──────────────────────────────────────────────────────────── training

@dataclass
class PlacementTrainConfig:
    """Recipe for the placement policy.

    The shipping pipeline is **scaled imitation learning** of the coach: collect a
    large pure-coach dataset and behaviour-clone it with a regularised CNN. This
    reaches ~79% held-out coach-accuracy and ~4.5 lines (vs 0 for random, ~8.7 for
    the coach). Held-out accuracy scales with data (32% @ 5k -> 57% @ 30k -> 79% @
    120k), so ``bc_states`` is the main quality knob.

    PPO fine-tuning is available (``ppo_updates`` > 0) but is **off by default**:
    empirically it degrades the imitation policy here, because on-policy RL flattens
    the sharp, near-optimal BC action distribution faster than the shaped reward can
    justify. See docs/ai_pipeline.md.
    """

    difficulty: Difficulty = NORMAL
    bc_states: int = 120000
    bc_epochs: int = 35
    dropout: float = 0.4
    weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    batch_size: int = 256
    # Optional DAgger refinement on top of BC (relabel learner-visited boards). 0 = skip.
    dagger_iterations: int = 0
    dagger_episodes: int = 60
    # Optional PPO fine-tune. 0 = skip (recommended; it degrades the BC policy here).
    ppo_updates: int = 0
    ppo_rollout_episodes: int = 16
    ppo_epochs: int = 4
    ppo_learning_rate: float = 8e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    value_weight: float = 0.5
    entropy_weight: float = 0.003
    kl_coef: float = 0.02
    max_pieces: int = 300
    eval_episodes: int = 10
    seed: int = 0
    checkpoint_path: Path = Path("artifacts/placement_policy.pt")


@dataclass
class PlacementSummary:
    eval_lines: list[float] = field(default_factory=list)
    best_eval_lines: float = 0.0
    checkpoint_path: Path | None = None
    phase_log: list[str] = field(default_factory=list)


def evaluate_placement(policy: PlacementPolicy, difficulty: Difficulty, episodes: int = 10, seed0: int = 7000) -> tuple[float, float]:
    """Return (mean_lines, game_over_rate) under greedy play."""
    env = PlacementEnv(difficulty=difficulty)
    total, overs = 0, 0
    for i in range(episodes):
        env.reset(seed=seed0 + i)
        while True:
            slot = policy.act(env.game_state, deterministic=True)
            if slot is None:
                break
            step = env.step(slot)
            if step.terminated:
                break
        total += env.game_state.lines_cleared
        overs += int(env.game_state.game_over)
    return total / episodes, overs / episodes


def _collect_dagger_batch(
    model: CNNActorCritic,
    difficulty: Difficulty,
    episodes: int,
    seed0: int,
    use_policy: bool,
    max_pieces: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Roll out (coach on iter 0, else the learner) and label every visited board with the coach."""
    policy = PlacementPolicy(model)
    planes_buf: list[np.ndarray] = []
    aux_buf: list[np.ndarray] = []
    labels: list[int] = []
    env = PlacementEnv(difficulty=difficulty, max_pieces=max_pieces)
    for i in range(episodes):
        env.reset(seed=seed0 + i)
        while True:
            expert = env.coach_slot()
            if expert is None:
                break
            planes, aux = observe(env.game_state)
            planes_buf.append(planes)
            aux_buf.append(aux)
            labels.append(expert)
            act = policy.act(env.game_state, deterministic=False) if use_policy else expert
            if act is None:
                break
            step = env.step(act)
            if step.terminated:
                break
    return planes_buf, aux_buf, labels


def _train_bc(model: CNNActorCritic, optimizer, planes, aux, labels, epochs: int, batch_size: int) -> float:
    planes_t = torch.as_tensor(np.array(planes))
    aux_t = torch.as_tensor(np.array(aux))
    labels_t = torch.tensor(labels, dtype=torch.long)
    n = len(labels_t)
    model.train()
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits, _ = model(planes_t[idx], aux_t[idx])
            loss = F.cross_entropy(logits, labels_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last = float(loss.item())
    model.eval()
    return last


def _collect_coach_dataset(
    difficulty: Difficulty, n_states: int, seed0: int, max_pieces: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Roll out the coach and record (board, aux, chosen_slot) for every visited state."""
    planes_buf: list[np.ndarray] = []
    aux_buf: list[np.ndarray] = []
    labels: list[int] = []
    env = PlacementEnv(difficulty=difficulty, max_pieces=max_pieces)
    episode = 0
    while len(planes_buf) < n_states:
        env.reset(seed=seed0 + episode)
        episode += 1
        while len(planes_buf) < n_states:
            slot = env.coach_slot()
            if slot is None:
                break
            planes, aux = observe(env.game_state)
            planes_buf.append(planes)
            aux_buf.append(aux)
            labels.append(slot)
            if env.step(slot).terminated:
                break
    return planes_buf, aux_buf, labels


def _placement_survival_reward(prev_lines: int, game_state: GameState, difficulty: Difficulty) -> float:
    """Survival-positive shaped reward for PPO.

    The default ``calculate_reward`` adds a height/bumpiness penalty per placement,
    so a *summed* return punishes survival (more pieces = more penalty) — which makes
    naive PPO learn to die sooner. This reward keeps each surviving placement net
    positive (alive bonus + line-clear bonus, light hole penalty, terminal penalty).
    """
    cleared = game_state.lines_cleared - prev_lines
    if game_state.game_over:
        return -3.0
    snapshot = build_snapshot("playing", game_state, difficulty, [EASY, NORMAL, HARD])
    holes = calculate_reward(snapshot, snapshot).profile.hole_count
    return 0.2 + 3.0 * cleared - 0.03 * holes


def _ppo_finetune(model: CNNActorCritic, config: PlacementTrainConfig, summary: PlacementSummary, verbose: bool) -> None:
    """Optional, KL-anchored PPO fine-tune that keeps the best-by-eval weights.

    NOTE: empirically this does not improve the imitation policy on this task (the
    sharp BC distribution gets flattened faster than the reward justifies). It is
    off by default (config.ppo_updates == 0) and kept for experimentation. Because it
    keeps the best-by-eval snapshot, enabling it can never *ship* a worse policy.
    """
    import copy

    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo_learning_rate)
    reference = copy.deepcopy(model)
    for param in reference.parameters():
        param.requires_grad_(False)

    best_state = copy.deepcopy(model.state_dict())
    env = PlacementEnv(difficulty=config.difficulty, max_pieces=config.max_pieces)
    seed_counter = 20000
    for update in range(config.ppo_updates):
        planes_b, aux_b, mask_b, act_b, logp_b, val_b, rew_b, done_b = [], [], [], [], [], [], [], []
        for _ in range(config.ppo_rollout_episodes):
            env.reset(seed=seed_counter)
            seed_counter += 1
            while True:
                mask_np = env.legal_mask()
                if not mask_np.any():
                    break
                planes, aux = observe(env.game_state)
                prev_lines = env.game_state.lines_cleared
                pt = torch.as_tensor(planes).unsqueeze(0)
                at = torch.as_tensor(aux).unsqueeze(0)
                mt = torch.as_tensor(mask_np).unsqueeze(0)
                with torch.no_grad():
                    logits, value = model(pt, at)
                    dist = Categorical(logits=_masked_logits(logits, mt))
                    action = dist.sample()
                step = env.step(int(action.item()))
                reward = _placement_survival_reward(prev_lines, env.game_state, config.difficulty)
                planes_b.append(planes); aux_b.append(aux); mask_b.append(mask_np)
                act_b.append(int(action.item()))
                logp_b.append(float(dist.log_prob(action).item()))
                val_b.append(float(value.item()))
                rew_b.append(reward)
                done_b.append(step.terminated)
                if step.terminated:
                    break
        if not act_b:
            continue
        returns, advs = [], []
        gae, next_val = 0.0, 0.0
        for t in reversed(range(len(rew_b))):
            mask = 0.0 if done_b[t] else 1.0
            delta = rew_b[t] + config.gamma * next_val * mask - val_b[t]
            gae = delta + config.gamma * config.gae_lambda * mask * gae
            advs.insert(0, gae)
            returns.insert(0, gae + val_b[t])
            next_val = val_b[t]
        planes_t = torch.as_tensor(np.array(planes_b))
        aux_t = torch.as_tensor(np.array(aux_b))
        mask_t = torch.as_tensor(np.array(mask_b))
        act_t = torch.tensor(act_b, dtype=torch.long)
        old_logp = torch.tensor(logp_b, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)
        adv_t = torch.tensor(advs, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(act_t)
        model.eval()  # dropout off so old/new log-probs match
        for _ in range(config.ppo_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, config.batch_size):
                idx = perm[start : start + config.batch_size]
                logits, values = model(planes_t[idx], aux_t[idx])
                dist = Categorical(logits=_masked_logits(logits, mask_t[idx]))
                new_logp = dist.log_prob(act_t[idx])
                ratio = torch.exp(new_logp - old_logp[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret_t[idx])
                entropy = dist.entropy().mean()
                with torch.no_grad():
                    ref_logits, _ = reference(planes_t[idx], aux_t[idx])
                    ref_dist = Categorical(logits=_masked_logits(ref_logits, mask_t[idx]))
                kl = torch.distributions.kl_divergence(ref_dist, dist).mean()
                loss = policy_loss + config.value_weight * value_loss - config.entropy_weight * entropy + config.kl_coef * kl
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if (update + 1) % 10 == 0:
            mean_lines, _ = evaluate_placement(PlacementPolicy(model), config.difficulty, episodes=config.eval_episodes)
            summary.eval_lines.append(mean_lines)
            if mean_lines > summary.best_eval_lines:
                summary.best_eval_lines = mean_lines
                best_state = copy.deepcopy(model.state_dict())
            msg = f"ppo update {update + 1}: greedy mean lines={mean_lines:.1f}"
            summary.phase_log.append(msg)
            if verbose:
                print(msg, flush=True)

    model.load_state_dict(best_state)  # never ship a worse policy than we started with


def train_placement_policy(config: PlacementTrainConfig, verbose: bool = False) -> PlacementSummary:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    model = CNNActorCritic(dropout=config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    summary = PlacementSummary()

    # ── Phase 1: scaled imitation of the coach (the part that works) ──
    planes, aux, labels = _collect_coach_dataset(config.difficulty, config.bc_states, seed0=0, max_pieces=config.max_pieces)
    agg_planes, agg_aux, agg_labels = list(planes), list(aux), list(labels)
    _train_bc(model, optimizer, agg_planes, agg_aux, agg_labels, config.bc_epochs, config.batch_size)
    mean_lines, _ = evaluate_placement(PlacementPolicy(model), config.difficulty, episodes=config.eval_episodes)
    summary.eval_lines.append(mean_lines)
    summary.best_eval_lines = max(summary.best_eval_lines, mean_lines)
    msg = f"bc: dataset={len(agg_labels)} greedy mean lines={mean_lines:.1f}"
    summary.phase_log.append(msg)
    if verbose:
        print(msg, flush=True)

    # ── Phase 2 (optional): DAgger refinement — relabel the learner's own boards ──
    for it in range(config.dagger_iterations):
        d_planes, d_aux, d_labels = _collect_dagger_batch(
            model, config.difficulty, episodes=config.dagger_episodes,
            seed0=500000 + it * 1000, use_policy=True, max_pieces=config.max_pieces,
        )
        agg_planes += d_planes
        agg_aux += d_aux
        agg_labels += d_labels
        _train_bc(model, optimizer, agg_planes, agg_aux, agg_labels, config.bc_epochs, config.batch_size)
        mean_lines, _ = evaluate_placement(PlacementPolicy(model), config.difficulty, episodes=config.eval_episodes)
        summary.eval_lines.append(mean_lines)
        summary.best_eval_lines = max(summary.best_eval_lines, mean_lines)
        msg = f"dagger iter {it + 1}/{config.dagger_iterations}: dataset={len(agg_labels)} greedy mean lines={mean_lines:.1f}"
        summary.phase_log.append(msg)
        if verbose:
            print(msg, flush=True)

    # ── Phase 3 (optional, off by default): PPO fine-tune ──
    if config.ppo_updates > 0:
        _ppo_finetune(model, config, summary, verbose)

    save_placement_checkpoint(config.checkpoint_path, model, config, summary)
    summary.checkpoint_path = config.checkpoint_path
    return summary


def save_placement_checkpoint(path: Path, model: CNNActorCritic, config: PlacementTrainConfig, summary: PlacementSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "cnn_actor_critic_placement",
            "n_slots": N_SLOTS,
            "difficulty": config.difficulty.name,
            "best_eval_lines": summary.best_eval_lines,
            "eval_lines": summary.eval_lines,
        },
        path,
    )


def load_placement_policy(path: Path, device: str | torch.device | None = None) -> PlacementPolicy:
    payload = torch.load(path, map_location=device or "cpu", weights_only=True)
    model = CNNActorCritic()
    model.load_state_dict(payload["model_state_dict"])
    return PlacementPolicy(model, device=device)
