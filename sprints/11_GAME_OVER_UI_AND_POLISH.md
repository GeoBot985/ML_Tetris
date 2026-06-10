# Sprint 11 — Game Over, HUD, and Polish

## Goal

Make the game presentable.

## Required Work

Add HUD display:

- Score
- Level
- Lines cleared
- Next piece preview
- Hold piece preview
- Pause state
- Game over message
- Restart instruction

Improve visual clarity:

- Grid lines if useful
- Consistent colors
- Window title
- Clean layout

## Required Tests

Mostly keep this sprint manual, but add tests for any new formatting/state helpers.

Optional tests:

1. HUD state object contains score, level, and lines.
2. Game over state is exposed cleanly.
3. Restart resets score, level, lines, board, hold, and queue.

## Acceptance Criteria

- Game communicates current state clearly.
- Game over is obvious.
- Restart works.
- Existing tests pass.
- UI polish does not introduce game-rule logic into renderer.

## Stop Point

Stop after the game is presentable and stable.
