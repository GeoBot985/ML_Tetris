# Architecture

## Overview

The project is split into two top-level packages under `src/`:

| Package | Role |
|---|---|
| `tetris` | Pure game logic and pygame presentation |
| `ai_agent` | RL environment, policy, training, evaluation, deployment |

The packages depend in one direction only: `ai_agent` imports from `tetris`, never the reverse. The pygame surface is never imported by `ai_agent`.

---

## tetris package

```
tetris/
├── board.py          Board — 10×20 grid, can_place, lock_piece, clear_rows
├── pieces.py         Piece dataclass, 7 shapes × 4 rotations (SRS cell offsets)
├── piece_source.py   seven_bag_piece_source() — fair piece distribution generator
├── game_state.py     GameState — active piece, queue, hold, gravity, scoring, game-over
├── commands.py       apply_command() — pygame-free command dispatcher (importable headlessly)
├── input_handler.py  pygame KEY_TO_COMMAND mapping + re-exports apply_command
├── difficulty.py     Difficulty dataclass — start_ms, per_level_ms, minimum_ms presets
├── scoring.py        line_clear_score, soft_drop_score, hard_drop_score
├── renderer.py       Renderer — board, ghost piece, sidebar, overlays, start screen
├── api.py            ApiBridge (queue), build_snapshot(), HTTP server (port 8765)
└── main.py           pygame event loop, launcher menu, subprocess job management
```

### Key invariants

- `GameState` is a plain Python object with no pygame dependency.
- `commands.py` is the canonical command dispatcher. `input_handler.py` imports and re-exports it for backward compatibility, adding only the pygame key mapping.
- `Board.lock_piece()` returns the number of cleared rows; `GameState.lock_active_piece()` uses this to update score and level.
- Gravity is purely time-based: `Difficulty.gravity_for_level(level) → int (ms)`. The pygame loop tracks elapsed ms and calls `gravity_tick()` when the threshold is reached.

---

## ai_agent package

```
ai_agent/
├── placement.py      Placement env + CNN policy + scaled-imitation trainer (the working AI stack)
├── environment.py    TetrisEnvironment (Gym-like), snapshot↔observation codec
├── policy.py         PPOAgentModel (MLP), PPOPolicy, PolicyDecision
├── rewards.py        profile_board(), calculate_reward(), BoardProfile, RewardBreakdown
├── safety.py         SafetyWrapper — legality, risk score, coach correction
├── training.py       train_policy(), coach_action(), compute_gae(), checkpoint IO
├── evaluation.py     evaluate_policy(), EvaluationResult, human baseline comparison
├── deployment.py     benchmark_policy(), ManifestLogger, build_manifest_record, quantisation
├── diagnostics.py    build_decision_diagnostics() — in-game AI overlay data
└── vectorized.py     make_vec_env() — subprocess-based parallel environments
```

### Data flow

```
GameState.board + active_piece
    → build_snapshot()               (tetris.api)
    → snapshot_to_observation()      (ai_agent.environment)
    → PPOAgentModel.forward()        (ai_agent.policy)
    → SafetyWrapper.decide()         (ai_agent.safety)
    → apply_command(game_state, …)   (tetris.commands)
    → calculate_reward(prev, next)   (ai_agent.rewards)
```

### Observation vector

`snapshot_to_observation()` produces a 600-dimensional `float32` vector:

| Slice | Content |
|---|---|
| `[0:200]` | Locked board (10×20, piece-type encoded as float) |
| `[200:400]` | Active piece mask (10×20 binary) |
| `[400:407]` | Active piece one-hot (7 piece types) |
| `[407]` | Active piece rotation (0–3, normalised) |
| `[408:415]` | Hold piece one-hot |
| `[415]` | Hold-used flag |
| `[416:451]` | Next queue one-hot (5 pieces × 7) |
| `[451:455]` | score, level, lines, gravity (all normalised) |
| `[455:457]` | paused, game_over flags |
| `[457:460]` | App-state one-hot (start / playing / game_over) |

---

## Thread model

The pygame loop runs on the main thread. The HTTP API server runs on a `ThreadingHTTPServer` in a daemon thread. Commands are passed through a `queue.Queue` in `ApiBridge` and drained once per frame by the main loop. No game state is mutated from the API thread.

---

## Headless vs. windowed

The `ai_agent` package can be imported in a process that never calls `pygame.init()`. The only entry point that starts a display is `tetris.main`. All `ai_agent` code uses `snapshot` dicts, not `GameState` objects, so it can run in isolated subprocesses for vectorised training.
