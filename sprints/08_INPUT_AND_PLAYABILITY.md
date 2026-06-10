# Sprint 08 — Input and Playability

## Goal

Make the game playable with keyboard input.

## Required Work

Create if useful:

```text
src/tetris/input_handler.py
```

Implement controls:

```text
Left arrow  = move left
Right arrow = move right
Down arrow  = soft drop
Space       = hard drop
Up arrow    = rotate clockwise
Z           = rotate counter-clockwise
P           = pause/unpause
R           = restart after game over
Esc         = quit
```

Implement timed gravity:

- Active piece falls automatically based on level.
- Higher levels should fall faster.

## Required Tests

Avoid testing pygame event internals heavily. Test the command mapping and game-state effects where possible.

Create/update:

```text
tests/test_input_handler.py
tests/test_game_state.py
```

Test:

1. Key mapping maps expected keys to commands.
2. Left command moves active piece left.
3. Right command moves active piece right.
4. Rotate command rotates active piece.
5. Hard drop locks piece.
6. Pause stops automatic gravity.
7. Unpause resumes gravity.
8. Restart creates a fresh game state after game over.

## Acceptance Criteria

- Game is playable.
- Existing tests pass.
- Input handling is separated from GameState rules.
- Pause and restart work.

## Stop Point

Stop after keyboard playability is working.
