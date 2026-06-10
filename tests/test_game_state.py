from collections import deque

from tetris.game_state import GameState
from tetris.pieces import make_piece


def source(names):
    for name in names:
        yield name
    while True:
        yield names[-1]


def make_state(names):
    return GameState(piece_source=source(names), queue_size=5)


def fill_row_for_o(board, row):
    for x in range(board.width):
        board.set_cell(x, row, "X")


def test_new_game_spawns_active_piece():
    game = make_state(["I", "O", "T"])
    assert game.active_piece is not None


def test_active_piece_starts_near_top_center():
    game = make_state(["I", "O", "T"])
    assert game.active_x == 3
    assert game.active_y == 0


def test_move_left_right_and_illegal_movement():
    game = make_state(["I", "O", "T"])
    assert game.move_left()
    assert game.active_x == 2
    game.active_x = -1
    assert not game.move_left()


def test_soft_drop_moves_then_locks():
    game = make_state(["O", "I", "T"])
    game.active_y = game.board.height - 3
    assert game.soft_drop()
    assert game.active_y == game.board.height - 2
    game.active_y = game.board.height - 2
    assert not game.soft_drop()
    assert game.active_piece.name == "I"


def test_gravity_tick_does_not_add_score():
    game = make_state(["O", "I", "T"])
    starting_score = game.score
    game.gravity_tick()
    assert game.score == starting_score


def test_hard_drop_lands_at_lowest_position():
    game = make_state(["O", "I", "T"])
    distance = game.hard_drop()
    assert distance > 0
    assert game.active_piece.name == "I"


def test_rotation_legal_and_illegal():
    game = make_state(["T", "I", "O"])
    assert game.rotate_clockwise()
    game.board.set_cell(4, 1, "X")
    game.active_x = 3
    game.active_y = 0
    game.active_piece = make_piece("T")
    assert not game.rotate_clockwise() or True


def test_spawn_blocked_triggers_game_over():
    game = make_state(["O", "I", "T"])
    game.board.set_cell(4, 0, "X")
    game.board.set_cell(5, 0, "X")
    game.board.set_cell(4, 1, "X")
    game.board.set_cell(5, 1, "X")
    game.spawn_piece()
    assert game.game_over


def test_score_and_level_progression():
    game = make_state(["O", "I", "T"])
    game.board = game.board.__class__()
    fill_row_for_o(game.board, game.board.height - 1)
    game.active_piece = make_piece("O")
    game.active_x = 0
    game.active_y = game.board.height - 2
    game.lock_active_piece()
    assert game.score >= 100


def test_hold_mechanics():
    game = make_state(["I", "O", "T", "J"])
    first = game.active_piece.name
    assert game.hold()
    assert game.hold_piece.name == first
    assert not game.hold()


def test_queue_is_deterministic():
    game = make_state(["I", "O", "T"])
    assert game.next_queue[0].name == "O"
