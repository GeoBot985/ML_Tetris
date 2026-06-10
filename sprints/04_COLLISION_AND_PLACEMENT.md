# Sprint 04 — Collision and Placement

## Goal

Connect pieces to the board through collision detection and placement.

No active game loop yet. No pygame.

## Required Work

Update board/game logic to support:

- Checking whether a piece can occupy a position
- Rejecting movement outside board bounds
- Rejecting movement into occupied cells
- Locking a piece onto the board
- Clearing lines after lock

You may create:

```text
src/tetris/game_state.py
```

if it helps keep board and active-piece behavior separate.

## Required Tests

Create or update:

```text
tests/test_game_state.py
tests/test_board.py
```

Test:

1. Piece can spawn on an empty board.
2. Piece cannot occupy negative x position.
3. Piece cannot occupy beyond right edge.
4. Piece cannot occupy below bottom.
5. Piece cannot overlap existing locked cells.
6. Locking a piece writes exactly four cells.
7. Locking a piece can trigger one line clear.
8. Locking a piece can trigger multiple line clears.
9. Line clear count is reported correctly.
10. Board remains valid after line clears.

## Acceptance Criteria

- All tests pass.
- Collision logic is deterministic.
- Placement logic does not depend on rendering.
- Board and piece responsibilities remain cleanly separated.

## Stop Point

Stop after collision and placement tests pass.
