import pygame

from tetris.input_handler import KEY_TO_COMMAND, apply_command, command_for_key
from tetris.game_state import GameState


def test_key_mapping():
    assert command_for_key(pygame.K_LEFT) == "left"
    assert KEY_TO_COMMAND[pygame.K_SPACE] == "hard_drop"
    assert command_for_key(pygame.K_z) == "rotate_cw"
    assert command_for_key(pygame.K_UP) == "rotate_cw"
    assert command_for_key(pygame.K_c) == "hold"


def test_commands_affect_game_state():
    def src():
        while True:
            yield "O"

    game = GameState(piece_source=src(), queue_size=5)
    x = game.active_x
    apply_command(game, "right")
    assert game.active_x == x + 1
    apply_command(game, "pause")
    assert game.paused
