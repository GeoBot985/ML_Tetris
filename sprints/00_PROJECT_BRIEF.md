# Sprint 00 — Project Brief and Working Rules

## Project

Build a Python Tetris clone with a disciplined test-first workflow.

The purpose of this project is to evaluate whether a local LLM, launched through Codex, can build a moderately complex application while maintaining architecture, regression tests, and clean incremental delivery.

## Core Rules

1. Do not build the full game in one pass.
2. Each sprint must be completed independently.
3. Each sprint must add or update tests before implementation.
4. All tests must pass before moving to the next sprint.
5. Keep the codebase small, readable, and modular.
6. Avoid unnecessary features unless explicitly requested by the sprint.
7. Prefer deterministic logic over UI-driven behavior.
8. Do not hide failing tests by weakening assertions.
9. Do not remove tests unless they are truly obsolete and replaced by better ones.
10. At the end of each sprint, provide:
    - Files changed
    - Tests added
    - Tests passing
    - Any known limitations

## Technology Baseline

Use Python.

Recommended stack:

- Python 3.11+
- pytest
- pygame for rendering and input

Do not introduce heavy frameworks.

## Expected Project Structure

```text
tetris_clone/
    pyproject.toml
    README.md
    src/
        tetris/
            __init__.py
            board.py
            pieces.py
            game_state.py
            scoring.py
            renderer.py
            input_handler.py
            main.py
    tests/
        test_board.py
        test_pieces.py
        test_game_state.py
        test_scoring.py
```

The exact structure may evolve, but the project must remain modular.

## Definition of Done for the Whole Project

The final app must include:

- Playable Tetris clone
- Board and piece logic
- Collision detection
- Piece movement
- Piece rotation
- Line clearing
- Scoring
- Level/speed progression
- Game over detection
- Next piece preview
- Hold piece
- Pause/restart
- Test suite covering core deterministic logic
- README with setup and run instructions
