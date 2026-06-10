# Sprint 03 — Piece Model

## Goal

Implement Tetris piece definitions and rotation logic.

No board collision yet. No pygame.

## Required Work

Create:

```text
src/tetris/pieces.py
```

Implement the seven standard tetrominoes:

- I
- O
- T
- S
- Z
- J
- L

Each piece must expose:

- type/name
- rotation state
- occupied cell offsets
- rotate clockwise
- rotate counter-clockwise
- reset or copy behavior if useful

The implementation may use matrices or coordinate offsets, but it must be deterministic and testable.

## Required Tests

Create:

```text
tests/test_pieces.py
```

Test:

1. All seven pieces exist.
2. Each piece has exactly four occupied blocks.
3. O piece rotation does not change shape.
4. Rotating a piece four times clockwise returns to original shape.
5. Rotating a piece clockwise then counter-clockwise returns to original shape.
6. Piece coordinates are stable and deterministic.
7. Creating two pieces of the same type does not share mutable rotation state.

## Acceptance Criteria

- All piece tests pass.
- No board or UI logic is mixed into piece definitions.
- Piece rotation is predictable and tested.

## Stop Point

Stop after piece model tests pass.
