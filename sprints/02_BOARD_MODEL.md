# Sprint 02 — Board Model

## Goal

Implement the Tetris board as a deterministic data model.

No pygame. No UI.

## Required Work

Create:

```text
src/tetris/board.py
```

Implement a `Board` class or equivalent with:

- width
- height
- grid
- empty cell representation
- bounds checking
- get cell
- set cell
- row fullness check
- clearing full rows

Default board size:

```text
width = 10
height = 20
```

## Required Tests

Create:

```text
tests/test_board.py
```

Test:

1. Default board size is 10 x 20.
2. New board is empty.
3. Setting and getting a cell works.
4. Out-of-bounds coordinates are rejected.
5. Full row detection works.
6. Non-full row detection works.
7. Clearing one full row works.
8. Clearing multiple full rows works.
9. Cleared rows are replaced with empty rows at the top.
10. Board height remains constant after clearing rows.

## Acceptance Criteria

- All board tests pass.
- Board logic contains no rendering or input code.
- Row clearing returns the number of cleared rows.

## Stop Point

Stop after board model tests pass.
