# Changelog

## [0.3.0] — 2026-06-09 — AI that actually plays

Root-caused why training "didn't work" and the AI played at ~1 line, then built the
stack that fixes it. Full analysis in [docs/ai_pipeline.md](docs/ai_pipeline.md).

### Diagnosis (all reproduced experimentally)
- **Reward-hack in the action space.** `restart`/`pause` were RL actions; `restart`
  wiped the board mid-episode with no penalty/termination, so a random agent reached
  game-over 0/5 times in 3000 steps and episodes never ended. The shipped checkpoint's
  200 episodes all had `game_over=False` and negative best reward.
- **Dead coach path.** `train_policy` only ran from-scratch micro-action PPO; the
  coach warm-start was never called.
- **Representation.** A flat MLP could not even imitate the coach (~40% accuracy) —
  Tetris placement is spatial.

### Fixes
- **Structural:** integer actions now decode via `API_ACTIONS` (not the misaligned
  `ACTION_NAMES`); the RL env ignores control commands (`restart`/`pause`/`start`/`quit`)
  so the agent can only play.
- **New working AI stack** (`ai_agent/placement.py`): placement action space (40 masked
  `(rotation, column)` slots), `CNNActorCritic` encoder, and scaled imitation (DAgger /
  behaviour cloning) of the heuristic coach. Held-out imitation accuracy scales with
  data (32% @ 5k → 57% @ 30k+reg → 79% @ 120k). The learned net plays ~4 lines vs ~0
  for the micro-action PPO and 0 for random.
- **Honest PPO finding:** a complete masked/GAE PPO is included but degrades the
  imitation policy here (across aggressive/conservative/KL-anchored variants); it is
  off by default and keeps the best-by-eval snapshot when enabled.
- **Tooling:** `train_placement_ai.py`, `compare_ai.py --placement-checkpoint`,
  `tests/test_ai_placement.py`, and updated README + ai_pipeline docs with real numbers.

## [0.2.0] — 2026-06-09

### Sprint 001 — Architecture cleanup
- **Added** `src/tetris/commands.py` — pure command dispatcher with no pygame dependency. Headless AI code can now import `apply_command` without triggering a pygame import.
- **Updated** `src/tetris/input_handler.py` — now only owns the pygame key mapping. `apply_command` is re-exported from `commands.py`. Added `X` key → `rotate_ccw` mapping.
- **Updated** `src/ai_agent/environment.py` — imports `apply_command` from `tetris.commands` (was `tetris.input_handler`). Removed dead `elif command == "hold"` branch; hold is now handled uniformly by `apply_command`.
- **Fixed** `SafetyWrapper.decide()` — was calling `predict_from_snapshot` and `act_from_snapshot` (identical methods) in sequence, making two neural-network inferences per decision. Now calls `predict_from_snapshot` once and reads `.action` directly.

### Sprint 002 — Game polish
- **Added** ghost piece — `Renderer._draw_ghost_piece()` renders the landing position of the active piece with semi-transparent alpha blending.
- **Improved** game-over screen — `_draw_game_over_overlay()` now shows final score, lines cleared, and level in a dedicated panel instead of a single-line message.
- **Improved** start screen — three-column layout: Mode / Difficulty / Controls. Controls column shows the full keyboard legend a new player needs.
- **Improved** sidebar — hold piece and next-piece queue now render actual piece shapes (mini pixel art) instead of text labels.
- **Added** `X` key → rotate counter-clockwise.
- **Added** difficulty progression tests — monotonic decrease, Hard ≤ Normal ≤ Easy at every level, level-up on 10 lines.

### Sprint 003 — AI evaluation evidence
- **Added** `compare_ai.py` — runs Random, Coach, PPO, and Guarded PPO side-by-side and prints a comparison table. Writes `artifacts/evaluation_result.json`.
- `RandomPolicy`, `CoachPolicy`, `PPOPolicyWrapper`, `GuardedPolicyWrapper` are defined inline in `compare_ai.py`.

### Sprint 004 — AI behavior and reward shaping
- **Extended** `BoardProfile` — added `aggregate_height` (sum of column heights), `bumpiness` (surface roughness), and `max_well_depth` (deepest isolated column).
- **Tuned** reward weights — hole penalty increased from 2.5 to 5.0 per cell; added bumpiness penalty (0.18) and well penalty (0.12); aggregate height penalty added (0.002).
- **Added** board profile regression tests in `tests/test_ai_rewards.py`.

### Sprint 005 — Test suite and regression harness
- **Added** `tests/conftest.py` — registers the `slow` pytest mark. CI command: `python -m pytest -m "not slow"`.
- **Marked** `@pytest.mark.slow` on tests that call `train_policy()`, parallel vectorised training, or multi-process environments.
- **Added** `tests/test_regression.py` — covers piece spawning, 7-bag balancing, hold rules, rotation legality, line clearing, scoring, level progression, headless command mapping via `commands.py`, observation encoding/decoding, single-call SafetyWrapper assertion, manifest logging, and benchmark output format.

### Sprint 006 — Packaging
- **Updated** `pyproject.toml` — project renamed to `tetris-rl-workbench`, version bumped to `0.2.0`, pinned dependency floor versions, added pytest `slow` marker declaration and `--tb=short` default.
- **Added** `requirements.txt`.

### Sprint 007 — Portfolio README
- **Rewrote** `README.md` — project title, description, architecture diagram, AI pipeline diagram, full command reference, evaluation output table, controls table, known limitations, roadmap.

### Sprint 008 — Final hardening
- **Added** `docs/architecture.md` — package layout, key invariants, data flow, thread model, headless vs. windowed separation.
- **Added** `docs/ai_pipeline.md` — training algorithm, `TrainingConfig` reference, reward component table, evaluation modes, manifest logging format.
- **Added** `docs/safety_wrapper.md` — decision flow diagram, legality-checking rules, risk scoring formula, `SafetyDecision` fields, tuning guidance.

---

## [0.1.0] — initial release

- Playable Tetris clone: board, pieces, scoring, 7-bag, hold, pause, restart.
- HTTP REST API bridge for external AI clients.
- Gym-like `TetrisEnvironment` with 600-dim observation vector.
- PPO actor-critic with coach-guided training loop.
- Basic evaluation against human baseline.
- `SafetyWrapper` (pre-fix — called policy twice per decision).
