from __future__ import annotations

import pygame

from .commands import apply_command  # re-exported for callers that import from here

KEY_TO_COMMAND = {
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
    pygame.K_DOWN: "soft_drop",
    pygame.K_SPACE: "hard_drop",
    pygame.K_z: "rotate_cw",
    pygame.K_UP: "rotate_cw",
    pygame.K_x: "rotate_ccw",
    pygame.K_c: "hold",
    pygame.K_p: "pause",
    pygame.K_r: "restart",
    pygame.K_ESCAPE: "quit",
}


def command_for_key(key: int) -> str | None:
    return KEY_TO_COMMAND.get(key)


__all__ = ["KEY_TO_COMMAND", "command_for_key", "apply_command"]
