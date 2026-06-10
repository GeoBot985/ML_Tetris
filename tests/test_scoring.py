from tetris.scoring import hard_drop_score, line_clear_score, soft_drop_score


def test_line_clear_scores():
    assert line_clear_score(1, 1) == 100
    assert line_clear_score(2, 1) == 300
    assert line_clear_score(3, 1) == 500
    assert line_clear_score(4, 1) == 800


def test_drop_scores():
    assert soft_drop_score(3) == 3
    assert hard_drop_score(4) == 8
