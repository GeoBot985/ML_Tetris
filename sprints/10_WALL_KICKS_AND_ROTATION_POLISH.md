# Sprint 10 — Wall Kicks and Rotation Polish

## Goal

Improve rotation behavior near walls and existing blocks.

This does not need to fully implement official SRS, but it must be predictable and tested.

## Required Work

Implement simple wall kick attempts when rotating:

Suggested kick offsets:

```text
(0, 0)
(-1, 0)
(1, 0)
(-2, 0)
(2, 0)
(0, -1)
```

Rotation should:

1. Try normal rotation.
2. Try kick offsets in order.
3. Apply the first legal result.
4. Reject rotation if all attempts fail.

## Required Tests

Update:

```text
tests/test_game_state.py
```

Test:

1. Piece near left wall can rotate if a kick makes it legal.
2. Piece near right wall can rotate if a kick makes it legal.
3. Rotation into occupied cells is rejected if no kick works.
4. Rotation uses the first legal kick.
5. Failed rotation leaves piece orientation and position unchanged.
6. Successful kicked rotation updates both orientation and position if needed.

## Acceptance Criteria

- Rotation feels reasonable during play.
- Tests lock down the expected kick behavior.
- Failed rotations do not corrupt state.
- Existing tests pass.

## Stop Point

Stop after wall kick behavior is implemented and tested.
