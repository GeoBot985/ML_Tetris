from tetris.pieces import Piece, all_piece_names, make_piece


def test_all_seven_pieces_exist():
    assert set(all_piece_names()) == {"I", "O", "T", "S", "Z", "J", "L"}


def test_each_piece_has_four_blocks():
    for name in all_piece_names():
        assert len(make_piece(name).cells()) == 4


def test_o_piece_rotation_is_stable():
    piece = make_piece("O")
    assert piece.cells() == piece.rotate_clockwise().cells()


def test_four_clockwise_rotations_return_to_original_shape():
    piece = make_piece("T")
    rotated = piece
    for _ in range(4):
        rotated = rotated.rotate_clockwise()
    assert rotated.cells() == piece.cells()


def test_clockwise_then_counter_clockwise_returns_to_original_shape():
    piece = make_piece("L")
    assert piece.rotate_clockwise().rotate_counter_clockwise().cells() == piece.cells()


def test_piece_coordinates_are_deterministic():
    assert make_piece("S").cells() == make_piece("S").cells()


def test_pieces_do_not_share_rotation_state():
    a = make_piece("J")
    b = make_piece("J")
    assert a.rotate_clockwise().cells() != b.cells()
