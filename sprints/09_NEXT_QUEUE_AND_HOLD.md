# Sprint 09 — Next Piece Queue and Hold Piece

## Goal

Add quality-of-life Tetris mechanics: next-piece queue and hold piece.

## Required Work

Implement:

- Next piece queue
- Display of upcoming pieces
- Hold piece
- Hold can only be used once per falling piece
- Held piece swaps with active piece
- If no held piece exists, active piece is held and next piece spawns

Use deterministic queue support for tests.

## Required Tests

Update:

```text
tests/test_game_state.py
```

Add tests:

1. New game has a next queue.
2. Spawning consumes from the queue.
3. Queue replenishes as pieces are consumed.
4. Hold stores the active piece if hold is empty.
5. Holding with an existing held piece swaps pieces.
6. Hold cannot be used twice before locking.
7. Hold becomes available again after lock.
8. Hold spawn collision triggers game over if applicable.
9. Next queue is deterministic under a test piece source.

## Acceptance Criteria

- Next piece preview renders.
- Hold piece renders.
- Hold behavior matches tests.
- All previous tests still pass.

## Stop Point

Stop after queue and hold behavior is implemented and tested.
