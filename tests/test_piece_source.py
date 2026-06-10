from tetris.game_state import GameState
from tetris.piece_source import classic_uniform_source, seven_bag_piece_source
from tetris.pieces import all_piece_names


def test_seven_bag_contains_all_unique_pieces_each_cycle():
    source = seven_bag_piece_source(shuffle=lambda bag: bag.reverse())
    first_bag = [next(source) for _ in range(7)]
    second_bag = [next(source) for _ in range(7)]

    expected = set(all_piece_names())
    assert set(first_bag) == expected
    assert set(second_bag) == expected
    assert len(first_bag) == len(expected)
    assert len(second_bag) == len(expected)


def test_seven_bag_repeats_without_missing_pieces():
    source = seven_bag_piece_source(shuffle=lambda bag: None)
    pieces = [next(source) for _ in range(14)]

    expected = list(all_piece_names())
    assert pieces[:7] == expected
    assert pieces[7:] == expected


def test_classic_uniform_source_is_reproducible_with_seed():
    first = classic_uniform_source(seed=123)
    second = classic_uniform_source(seed=123)

    assert [next(first) for _ in range(20)] == [next(second) for _ in range(20)]


def test_classic_uniform_source_is_not_bag_constrained():
    source = classic_uniform_source(seed=0)
    pieces = [next(source) for _ in range(14)]

    assert all(piece in all_piece_names() for piece in pieces)
    assert any(left == right for left, right in zip(pieces, pieces[1:]))


def test_game_state_uses_seven_bag_by_default(monkeypatch):
    monkeypatch.setattr("tetris.piece_source.random.shuffle", lambda bag: bag.reverse())
    game = GameState(queue_size=7)
    expected = list(all_piece_names())
    expected.reverse()
    observed = [game.active_piece.name] + [piece.name for piece in list(game.next_queue)[:6]]
    assert observed == expected
