from tetris.board import Board
from tetris.pieces import make_piece


def test_default_dimensions():
    board = Board()
    assert board.width == 10
    assert board.height == 20


def test_initial_state():
    board = Board()
    for y in range(20):
        for x in range(10):
            assert board.get_cell(x, y) is None


def test_set_and_get_cell():
    board = Board()
    board.set_cell(5, 5, "block")
    assert board.get_cell(5, 5) == "block"


def test_out_of_bounds():
    board = Board()
    for coords in [(-1, 5), (10, 10)]:
        x, y = coords
        try:
            board.get_cell(x, y)
            assert False
        except ValueError:
            pass


def test_row_full_detection():
    board = Board()
    for x in range(10):
        board.set_cell(x, 5, "block")
    assert board.is_row_full(5)
    assert not board.is_row_full(4)


def test_row_not_full():
    board = Board()
    for x in range(9):
        board.set_cell(x, 2, "block")
    assert not board.is_row_full(2)


def test_clear_single_row_places_empty_row_at_top():
    board = Board()
    for x in range(10):
        board.set_cell(x, 10, "block")
    board.set_cell(0, 9, "above")

    cleared = board.clear_rows()

    assert cleared == 1
    assert len(board.grid) == 20
    assert board.get_cell(0, 0) is None
    assert board.get_cell(0, 10) == "above"


def test_clear_multiple_rows_places_empty_rows_at_top():
    board = Board()
    for x in range(10):
        board.set_cell(x, 10, "block")
        board.set_cell(x, 11, "block")
    board.set_cell(0, 9, "above")

    cleared = board.clear_rows()

    assert cleared == 2
    assert len(board.grid) == 20
    assert board.get_cell(0, 0) is None
    assert board.get_cell(0, 1) is None
    assert board.get_cell(0, 11) == "above"


def test_can_place_and_lock_piece():
    board = Board()
    piece = make_piece("O")

    assert board.can_place(piece.cells(), 3, 0)
    cleared = board.lock_piece(piece.cells(), 3, 0, piece.name)
    assert cleared == 0
    assert board.get_cell(4, 0) == "O"
    assert board.get_cell(5, 1) == "O"
