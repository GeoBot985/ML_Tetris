# Sprint 12 — Regression Hardening and README

## Goal

Harden the project as a finished local LLM coding benchmark.

## Required Work

Review the full codebase.

Improve:

- Naming
- Module boundaries
- Error handling
- Test coverage
- README
- Known limitations
- Run instructions

Add a final regression suite covering complete mini-scenarios.

## Required Tests

Add scenario-style tests:

1. Hard drop locks a piece and spawns the next piece.
2. Clearing four lines scores correctly.
3. Hold cannot be reused before lock.
4. Game over occurs when spawn area is blocked.
5. Restart returns the game to a clean initial state.
6. A sequence of moves produces expected board state.
7. A sequence involving line clears, score, and level progression behaves correctly.

## Documentation Requirements

README must include:

- Project summary
- Why the project exists
- Installation
- Running tests
- Running the game
- Controls
- Architecture overview
- Validation approach
- Known limitations
- Suggested future improvements

## Final Acceptance Criteria

- All tests pass.
- Game runs locally.
- Project is understandable from README alone.
- No major file is unnecessarily large.
- Core logic remains testable without pygame.
- The project can serve as a benchmark for local LLM coding discipline.

## Stop Point

Stop after final review and documentation are complete.
