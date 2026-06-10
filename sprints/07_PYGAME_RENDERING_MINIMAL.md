# Sprint 07 — Minimal Pygame Rendering

## Goal

Add a minimal playable window that renders the board and active piece.

This sprint introduces pygame, but must not weaken the deterministic test suite.

## Required Work

Create:

```text
src/tetris/renderer.py
src/tetris/main.py
```

Implement:

- Pygame window
- Board rendering
- Active piece rendering
- Basic colors per piece type
- Game loop shell
- Clock/tick handling
- Quit handling

The app should launch with:

```bash
python -m tetris.main
```

## Required Tests

Rendering is difficult to test deeply. Add lightweight tests only where practical.

Create:

```text
tests/test_renderer.py
```

Test:

1. Renderer can be constructed.
2. Board-to-screen coordinate conversion works.
3. Renderer exposes expected cell size and board dimensions.

Do not require graphical display in CI if possible. Avoid brittle screenshot tests.

## Acceptance Criteria

- Existing deterministic tests still pass.
- `python -m tetris.main` opens a window.
- Board and active piece are visible.
- Closing the window exits cleanly.
- Rendering code does not contain game rules.

## Stop Point

Stop after the minimal window works and all tests pass.
