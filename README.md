# Tetris RL Workbench

A Python Tetris clone extended into a **reinforcement-learning workbench** — PPO training, safety-guarded AI play, structured evaluation, inference benchmarking, and manifest-based decision logging.

---

## Why this project exists

Most RL toy projects either show a game _or_ show an RL agent. This one shows both running together with a clean separation:

- The game is a proper playable Tetris clone with keyboard controls, ghost piece, hold, and difficulty curves.
- The AI side is a gym-style environment (`TetrisEnvironment`) wrapping the same game logic headlessly.
- A `SafetyWrapper` guards the trained PPO policy against illegal moves and high-risk board states.
- A heuristic coach policy generates supervised warm-start data and serves as a comparison baseline.
- Every evaluation run produces structured JSON that can be pasted into a README table.

---

## Features

| Area | What's included |
|---|---|
| Game | 7-bag randomiser, SRS-style rotation + wall kicks, ghost piece, hold, 3-difficulty speed curve |
| AI environments | Gym-like micro-action env (600-dim vector) **and** a placement env (40 masked slots) |
| Learned policy | **CNN placement policy** trained by scaled imitation of a heuristic coach (~4 lines, ~20× the micro-action PPO) |
| RL | Masked, GAE-based PPO for both action spaces (with an honest finding: imitation beats PPO on this task — see docs) |
| Safety | `SafetyWrapper` — legality checking, risk scoring (height + holes + critic value), coach correction |
| Evaluation | Compare Random / Coach / PPO / Guarded PPO / CNN-Placement in one command |
| Benchmarking | Per-decision latency (mean + P95) with optional INT8 quantisation |
| Logging | JSONL manifest with per-decision state hash, latency, correction reason |

---

## Quick start

```bash
# 1 — install
pip install -r requirements.txt

# 2 — play the game
python -m tetris.main

# 3 — train the WORKING learned policy (CNN placement, imitation of the coach)
python train_placement_ai.py --difficulty normal

# 3b — (legacy/experiment) micro-action PPO — plays poorly by design, see docs
python train_ai.py --difficulty normal --episodes 200 --checkpoint artifacts/ai_policy.pt

# 4 — compare all policies
python compare_ai.py --difficulty normal --episodes 10 \
    --checkpoint artifacts/ai_policy.pt \
    --placement-checkpoint artifacts/placement_policy.pt

# 5 — formal evaluation against human baseline
python evaluate_ai.py --checkpoint artifacts/ai_policy.pt --difficulty normal --episodes 20 \
       --output-json artifacts/evaluation_result.json

# 6 — measure inference latency
python benchmark_ai.py --checkpoint artifacts/ai_policy.pt --difficulty normal \
       --output-json artifacts/benchmark_result.json

# 7 — watch the agent play in the game window
python run_ai.py --render --policy ppo --checkpoint artifacts/ai_policy.pt
```

---

## Controls

| Key | Action |
|---|---|
| `← / →` | Move piece |
| `↓` | Soft drop |
| `Space` | Hard drop |
| `Z` / `↑` | Rotate clockwise |
| `X` | Rotate counter-clockwise |
| `C` | Hold |
| `P` | Pause |
| `R` | Restart |
| `Esc` | Quit |

On the launcher screen: `←/→` cycles mode, `↑/↓` selects difficulty, `Enter` launches.

---

## Architecture

```
src/
├── tetris/
│   ├── board.py          10×20 grid — collision, lock, line-clear
│   ├── pieces.py         7 tetrominoes, 4 rotations each, SRS cells
│   ├── piece_source.py   7-bag randomiser (generator)
│   ├── game_state.py     Active piece, queue, hold, gravity, scoring
│   ├── commands.py       Pure command dispatch (no pygame — importable headlessly)
│   ├── input_handler.py  pygame key → command string mapping
│   ├── difficulty.py     Speed curves: Easy / Normal / Hard
│   ├── scoring.py        Line-clear / soft-drop / hard-drop scores
│   ├── renderer.py       pygame rendering — board, ghost piece, sidebar, overlays
│   ├── api.py            HTTP REST bridge (port 8765) for external AI clients
│   └── main.py           pygame event loop + launcher menu
│
└── ai_agent/
    ├── placement.py      ★ WORKING STACK: placement env + CNN policy + scaled imitation (+ optional PPO)
    ├── environment.py    Gym-like micro-action TetrisEnvironment, snapshot↔observation codec
    ├── policy.py         PPOAgentModel (actor-critic MLP) + PPOPolicy  (micro-action / legacy)
    ├── rewards.py        BoardProfile metrics + shaped reward breakdown
    ├── safety.py         SafetyWrapper — legality check, risk score, coach correction
    ├── training.py       Micro-action PPO loop, GAE, coach_action heuristic, checkpoint IO
    ├── evaluation.py     Policy evaluation against human baselines
    ├── deployment.py     INT8 quantisation, latency benchmarking, ManifestLogger
    ├── diagnostics.py    Decision diagnostics for the in-game AI overlay
    └── vectorized.py     Parallel environment runner (subprocess vectorisation)

compare_ai.py             Side-by-side policy comparison (Random/Coach/PPO/Guarded)
train_ai.py               CLI entry point for training
evaluate_ai.py            CLI entry point for evaluation
benchmark_ai.py           CLI entry point for latency benchmarking
run_ai.py                 CLI entry point for headless or rendered AI play
```

---

## AI pipeline

```
Snapshot (game state JSON)
        │
        ▼
snapshot_to_observation()          ← 600-dim float32 numpy vector
        │
        ▼
PPOAgentModel.forward()            ← actor logits + critic value
        │
        ▼
SafetyWrapper.decide()
  ├─ is_legal_action()?            ← collision-checked legality test
  ├─ risk_score()?                 ← height + holes + critic penalty
  └─ fallback: coach_action()      ← beam-search heuristic placement scorer
        │
        ▼
executed_action → TetrisEnvironment.step()
        │
        ▼
calculate_reward()                 ← line clears, holes, bumpiness, wells, height
        │
        ▼
PPO update (rollout buffer → GAE → mini-batch gradient step)
```

---

## Safety wrapper

`SafetyWrapper` wraps any `PPOPolicy` and enforces two layers of safety:

1. **Legality**: Moves that would immediately be blocked by the board are rejected. Rotation is tested against all 6 SRS kick offsets.
2. **Risk**: If the critic value is very negative _and_ the board is tall and full of holes, the model's action is replaced by the heuristic coach.

The correction reason (`illegal_action`, `high_risk`, `safety_fallback`) is recorded in the manifest for later auditing.

---

## Reward shaping

| Component | Weight | Signal |
|---|---|---|
| Line clear | 1.0 – 10.0 | 1/2/3/4 lines cleared (Tetris bonus) |
| Complexity bonus | 0.25 – 4.0 | Combos and Tetrises rewarded extra |
| Stack height | 0.04/cell | Max column height |
| Aggregate height | 0.002/cell | Sum of all column heights |
| Holes | 5.0/cell | Cells below the surface but empty |
| Bumpiness | 0.18/step | Abs height difference between neighbours |
| Wells | 0.12/cell | Depth of deepest isolated column |
| Game over | −15.0 | Terminal penalty |

---

## Artifacts

After running training and evaluation the `artifacts/` folder will contain:

```
artifacts/
  ai_policy.pt              trained PPO checkpoint (torch state dict + metadata)
  training_metrics.jsonl    per-episode metrics (reward, lines, loss)
  training_feedback.md      human-readable training summary
  evaluation_result.json    policy comparison table (JSON)
  benchmark_result.json     latency stats (raw + guarded, mean + P95)
  manifest_sample.jsonl     per-decision log with state hash and correction reason
```

---

## Example evaluation output

Real numbers from `compare_ai.py` on Normal (5 episodes, seeds 1–5):

```
Policy                   Mean lines   Mean score  Game-over %   Safety corrections
----------------------------------------------------------------------------------
Random                          0.0          129         100%                  n/a
Coach (heuristic)              17.4         3850          40%                  n/a
Micro-action PPO (legacy)       0.2          316         100%                  n/a
Guarded PPO (legacy)            0.2          318         100%                 1.9%
CNN Placement (imitation)       4.2          420         100%                  n/a
```

Two AI stacks are shown deliberately (see [docs/ai_pipeline.md](docs/ai_pipeline.md)):

- **Micro-action PPO** (the original design) plays at ~0 lines. Trained from
  scratch over a keypress action space with a flat MLP, it cannot learn Tetris —
  a documented cautionary result, with the root-cause analysis in the docs.
- **CNN Placement** (the working design) plays at ~4 lines, ~20× better. It uses a
  placement action space (`place a piece`, 40 masked slots), a convolutional
  encoder, and scaled imitation of the heuristic coach. Held-out coach-imitation
  accuracy scales with data (32% @ 5k → 79% @ 120k samples).

Reproduce with:

```bash
python train_placement_ai.py --difficulty normal
python compare_ai.py --difficulty normal --episodes 10 \
    --checkpoint artifacts/ai_policy.pt \
    --placement-checkpoint artifacts/placement_policy.pt
```

---

## Running tests

```bash
# Fast tests only (no training) — CI-friendly
python -m pytest -m "not slow"

# Full suite including training smoke tests
python -m pytest

# Specific area
python -m pytest tests/test_regression.py -v
```

---

## HTTP API

The game window exposes a REST API on `http://127.0.0.1:8765` for external AI clients:

```bash
curl http://127.0.0.1:8765/state
curl -X POST http://127.0.0.1:8765/command -d '{"command":"left"}' -H 'Content-Type: application/json'
```

Available commands: `start left right soft_drop hard_drop rotate_cw rotate_ccw hold pause restart quit`

---

## Known limitations

- **The learned policy (~4 lines) does not match the heuristic coach (~17 lines).**
  It is limited by imitation accuracy (79% held-out); closing the gap needs more
  data and/or a stronger expert. Honest and reproducible — not world-class.
- **PPO does not improve the policy on this task.** Across aggressive, conservative,
  and KL-anchored variants it degrades the imitation policy (the sharp BC
  distribution flattens faster than the shaped reward justifies). Off by default;
  see [docs/ai_pipeline.md](docs/ai_pipeline.md).
- The micro-action PPO stack is kept only as a documented cautionary experiment.
- The ghost piece uses `SRCALPHA` blending which may be slow on some SDL backends.
- No T-spin or B2B detection — scoring follows the classic Tetris line-clear table.

---

## Roadmap

- [ ] Left/right symmetry augmentation to lift imitation accuracy (Tetris is mirror-symmetric)
- [ ] Scale the coach dataset to 250k+ and add a stronger lookahead coach
- [ ] Make PPO actually help: trust-region fine-tune from a high-accuracy BC start
- [ ] ONNX export for the trained policy
- [ ] Web viewer for manifest playback
