from __future__ import annotations

"""Pure command dispatch: no pygame dependency. Import this for headless/AI use."""

COMMAND_NAMES = frozenset(
    {"left", "right", "soft_drop", "hard_drop", "rotate_cw", "rotate_ccw", "hold", "pause", "restart", "quit"}
)


def apply_command(game_state, command: str):
    """Apply *command* to *game_state*. Returns "quit" for quit, else the GameState method return value."""
    if command == "left":
        return game_state.move_left()
    if command == "right":
        return game_state.move_right()
    if command == "soft_drop":
        return game_state.soft_drop()
    if command == "hard_drop":
        return game_state.hard_drop()
    if command == "rotate_cw":
        return game_state.rotate_clockwise()
    if command == "rotate_ccw":
        return game_state.rotate_counter_clockwise()
    if command == "hold":
        return game_state.hold()
    if command == "pause":
        return game_state.toggle_pause()
    if command == "restart":
        return game_state.restart()
    if command == "quit":
        return "quit"
    return None
