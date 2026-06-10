# Sprint 01 — Project Bootstrap

## Goal

Create the initial project skeleton, package structure, test setup, and minimal documentation.

Do not implement gameplay yet.

## Required Work

Create:

```text
pyproject.toml
README.md
src/tetris/__init__.py
tests/
```

Configure pytest so tests can be run with:

```bash
pytest
```

Add a trivial smoke test confirming the package imports.

## Required Tests

Create:

```text
tests/test_imports.py
```

Test:

- `import tetris` works.

## Acceptance Criteria

- `pytest` runs successfully.
- Project has a clear README.
- README includes:
  - Project purpose
  - Setup instructions
  - Test command
  - Run command placeholder
- No gameplay logic is implemented yet.

## Stop Point

Stop after the project skeleton and smoke test are working.
