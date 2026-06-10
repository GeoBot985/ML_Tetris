import pygame

from tetris.difficulty import NORMAL
from tetris.renderer import Renderer
from tetris.main import snap_size_to_aspect


def test_renderer_construction():
    renderer = Renderer()
    assert renderer.board_width == 10
    assert renderer.board_height == 20


def test_board_to_screen_conversion():
    renderer = Renderer()
    assert renderer.board_to_screen(0, 0) == (20, 20)


def test_renderer_exposes_expected_dimensions():
    renderer = Renderer(cell_size=24)
    assert renderer.cell_size == 24
    assert renderer.surface_width > 20 * 2 + 10 * 24


def test_start_screen_render_method_runs():
    pygame.init()
    try:
        renderer = Renderer()
        surface = pygame.Surface((renderer.surface_width, renderer.surface_height))
        renderer.draw_start_screen(surface, NORMAL)
    finally:
        pygame.quit()


def test_sidebar_and_overlay_draw_path_runs():
    from tetris.difficulty import NORMAL
    from tetris.game_state import GameState

    pygame.init()
    try:
        renderer = Renderer()
        surface = pygame.Surface((renderer.surface_width, renderer.surface_height))
        game = GameState(difficulty=NORMAL)
        game.paused = True
        renderer.draw(surface, game)
        game.paused = False
        game.game_over = True
        renderer.draw(surface, game)
    finally:
        pygame.quit()


def test_snap_size_to_aspect_preserves_ratio():
    width, height = snap_size_to_aspect(2000, 900, 16 / 9)
    assert abs((width / height) - (16 / 9)) < 0.01
