# Sprint 05 — Active Game State

## Goal

Implement the active falling-piece state machine.

No pygame yet.

## Required Work

Implement a `GameState` or equivalent that tracks:

- Board
- Active piece
- Active piece position
- Piece spawn
- Move left
- Move right
- Soft drop
- Hard drop
- Rotate active piece
- Lock active piece when it can no longer fall
- Spawn next piece
- Game over when spawning is blocked

Use a deterministic piece source for tests. Avoid random behavior in tests.

## Required Tests

Update:

```text
tests/test_game_state.py
```

Test:

1. New game spawns an active piece.
2. Active piece starts near top center.
3. Move left changes position when legal.
4. Move right changes position when legal.
5. Illegal movement is rejected.
6. Soft drop moves the piece down when possible.
7. Soft drop locks the piece when it cannot move down.
8. Hard drop moves the piece to the lowest legal position.
9. After lock, a new piece spawns.
10. Game over is triggered if spawn position is blocked.
11. Rotation is rejected if it would collide.
12. Rotation succeeds when legal.

## Acceptance Criteria

- All tests pass.
- Tests use deterministic piece order.
- No pygame or real-time loop is introduced yet.
- Game state can be stepped manually.

## Stop Point

Stop after active game state tests pass.
