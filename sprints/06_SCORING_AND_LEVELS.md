# Sprint 06 — Scoring and Levels

## Goal

Add scoring, line counts, and level/speed progression.

No pygame yet.

## Required Work

Create:

```text
src/tetris/scoring.py
```

Implement standard-ish scoring:

```text
Single line: 100 x level
Double line: 300 x level
Triple line: 500 x level
Tetris:      800 x level
Soft drop:  1 point per cell
Hard drop:  2 points per cell
```

Level progression:

```text
Start level: 1
Increase level every 10 cleared lines
```

GameState must track:

- score
- level
- total lines cleared

## Required Tests

Create:

```text
tests/test_scoring.py
```

Update:

```text
tests/test_game_state.py
```

Test:

1. Initial score is 0.
2. Initial level is 1.
3. Clearing one line scores 100 at level 1.
4. Clearing two lines scores 300 at level 1.
5. Clearing three lines scores 500 at level 1.
6. Clearing four lines scores 800 at level 1.
7. Level increases after 10 total lines.
8. Score uses current level multiplier.
9. Soft drop adds points.
10. Hard drop adds points based on dropped distance.
11. Score accumulates across multiple locks.

## Acceptance Criteria

- All tests pass.
- Scoring logic is isolated and testable.
- GameState integrates scoring cleanly.
- No UI code yet.

## Stop Point

Stop after scoring and level tests pass.
