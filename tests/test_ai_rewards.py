from ai_agent.rewards import calculate_reward, profile_board


def empty_board(width=4, height=4):
    return [[None for _ in range(width)] for _ in range(height)]


def snapshot(board, lines_cleared=0, paused=False, game_over=False):
    return {
        "locked_board": board,
        "lines_cleared": lines_cleared,
        "paused": paused,
        "game_over": game_over,
    }


def test_profile_board_detects_stack_height_and_holes():
    board = empty_board()
    board[1][0] = "X"
    board[3][0] = "X"

    profile = profile_board(snapshot(board))

    assert profile.stack_height == 3
    assert profile.hole_count == 1
    assert profile.row_hole_count == 1
    assert profile.hole_density > 0


def test_reward_increases_for_line_clears():
    previous = snapshot(empty_board())
    current = snapshot(empty_board(), lines_cleared=2)

    reward = calculate_reward(previous, current)

    assert reward.lines_cleared_delta == 2
    assert reward.line_clear_reward > 0
    assert reward.line_clear_complexity_reward > 0
    assert reward.total > 0


def test_tetris_clear_gets_more_complexity_reward_than_single_clear():
    previous = snapshot(empty_board())
    single_clear = calculate_reward(previous, snapshot(empty_board(), lines_cleared=1))
    tetris_clear = calculate_reward(previous, snapshot(empty_board(), lines_cleared=4))

    assert tetris_clear.line_clear_complexity_reward > single_clear.line_clear_complexity_reward
    assert tetris_clear.total > single_clear.total


def test_reward_penalizes_height_and_holes():
    previous = snapshot(empty_board())

    flat_board = empty_board()
    flat_board[3] = ["X", "X", "X", "X"]

    hole_board = empty_board()
    hole_board[1][0] = "X"
    hole_board[3][0] = "X"

    flat_reward = calculate_reward(previous, snapshot(flat_board))
    hole_reward = calculate_reward(previous, snapshot(hole_board))

    assert hole_reward.hole_penalty > flat_reward.hole_penalty
    assert hole_reward.row_hole_penalty > flat_reward.row_hole_penalty
    assert hole_reward.total < flat_reward.total


def test_reward_adds_runaway_penalty_at_five_rows():
    previous = snapshot(empty_board(4, 8))

    four_high = empty_board(4, 8)
    for row in range(4, 8):
        four_high[row][0] = "X"

    five_high = empty_board(4, 8)
    for row in range(3, 8):
        five_high[row][0] = "X"

    four_reward = calculate_reward(previous, snapshot(four_high))
    five_reward = calculate_reward(previous, snapshot(five_high))

    assert four_reward.profile.stack_height == 4
    assert five_reward.profile.stack_height == 5
    assert four_reward.stack_risk_penalty == 0
    assert five_reward.stack_risk_penalty > 0
    assert five_reward.total < four_reward.total


def test_reward_increases_when_stack_height_is_reduced():
    previous_board = empty_board(4, 8)
    for row in range(2, 8):
        previous_board[row][0] = "X"

    unchanged_board = empty_board(4, 8)
    for row in range(2, 8):
        unchanged_board[row][0] = "X"

    reduced_board = empty_board(4, 8)
    for row in range(4, 8):
        reduced_board[row][0] = "X"

    unchanged_reward = calculate_reward(snapshot(previous_board), snapshot(unchanged_board))
    reduced_reward = calculate_reward(snapshot(previous_board), snapshot(reduced_board))

    assert unchanged_reward.profile.stack_height == 6
    assert reduced_reward.profile.stack_height == 4
    assert unchanged_reward.stack_reduction_reward == 0
    assert reduced_reward.stack_reduction_reward > 0
    assert reduced_reward.total > unchanged_reward.total


def test_reward_penalizes_rows_left_with_holes():
    previous = snapshot(empty_board())

    clean_board = empty_board(4, 4)
    clean_board[1][0] = "X"
    clean_board[2][0] = "X"
    clean_board[3][0] = "X"

    row_hole_board = empty_board(4, 4)
    row_hole_board[1][0] = "X"
    row_hole_board[1][2] = "X"
    row_hole_board[3][0] = "X"
    row_hole_board[3][2] = "X"

    clean_reward = calculate_reward(previous, snapshot(clean_board))
    row_hole_reward = calculate_reward(previous, snapshot(row_hole_board))

    assert row_hole_reward.profile.row_hole_count == 1
    assert row_hole_reward.row_hole_penalty > clean_reward.row_hole_penalty
    assert row_hole_reward.total < clean_reward.total


# ── BoardProfile extended metrics ──────────────────────────────────────────

def test_profile_aggregate_height():
    # 4-wide, 4-tall board. Column 0: block at row 2 → height=2. Rest empty.
    board = empty_board(4, 4)
    board[2][0] = "X"
    p = profile_board(snapshot(board))
    assert p.aggregate_height == 2
    assert p.stack_height == 2


def test_profile_bumpiness_flat_board():
    board = empty_board(4, 4)
    board[3] = ["X", "X", "X", "X"]
    p = profile_board(snapshot(board))
    assert p.bumpiness == 0


def test_profile_bumpiness_uneven():
    # Col heights: [2, 0, 0, 0] → bumpiness = |2-0|+|0-0|+|0-0| = 2
    board = empty_board(4, 4)
    board[2][0] = "X"
    board[3][0] = "X"
    p = profile_board(snapshot(board))
    assert p.bumpiness == 2 + 0 + 0


def test_profile_well_depth_detected():
    # Col heights: [3, 1, 3] → col 1 is a well of depth 2
    board = empty_board(3, 4)
    for row in [1, 2, 3]:
        board[row][0] = "X"
    board[3][1] = "X"
    for row in [1, 2, 3]:
        board[row][2] = "X"
    p = profile_board(snapshot(board))
    assert p.max_well_depth >= 2


def test_profile_empty_board_all_zero():
    p = profile_board(snapshot(empty_board(4, 4)))
    assert p.stack_height == 0
    assert p.aggregate_height == 0
    assert p.hole_count == 0
    assert p.row_hole_count == 0
    assert p.bumpiness == 0
    assert p.max_well_depth == 0


def test_reward_breakdown_has_bumpiness_penalty():
    board = empty_board(4, 4)
    board[2][0] = "X"  # uneven surface
    r = calculate_reward(snapshot(empty_board(4, 4)), snapshot(board))
    assert r.bumpiness_penalty >= 0


def test_game_over_penalty_dominates():
    previous = snapshot(empty_board())
    current = snapshot(empty_board(), game_over=True)
    r = calculate_reward(previous, current)
    assert r.game_over_penalty > 0
    assert r.total < 0
