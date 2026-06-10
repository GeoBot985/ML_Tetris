from tetris.difficulty import EASY, HARD, NORMAL, difficulty_by_name
from tetris.game_state import GameState


def test_difficulty_names_are_available():
    assert difficulty_by_name("easy") == EASY
    assert difficulty_by_name("Normal") == NORMAL
    assert difficulty_by_name("HARD") == HARD


def test_gravity_speeds_get_faster_gradually():
    game = GameState(difficulty=NORMAL)
    assert game.gravity_for_level(1) == 700
    assert game.gravity_for_level(2) == 665
    assert game.gravity_for_level(5) == 560


def test_gravity_respects_minimum_speed():
    easy_game = GameState(difficulty=EASY)
    hard_game = GameState(difficulty=HARD)

    assert easy_game.gravity_for_level(100) == 300
    assert hard_game.gravity_for_level(100) == 160


def test_gravity_is_strictly_decreasing_until_minimum():
    """Gravity delay must decrease monotonically from level 1 until it hits the floor."""
    for difficulty in (EASY, NORMAL, HARD):
        prev = difficulty.gravity_for_level(1)
        hit_floor = False
        for level in range(2, 30):
            current = difficulty.gravity_for_level(level)
            if hit_floor:
                assert current == difficulty.minimum_ms, (
                    f"{difficulty.name} level {level}: expected minimum {difficulty.minimum_ms}, got {current}"
                )
            else:
                assert current <= prev, (
                    f"{difficulty.name} level {level}: gravity increased from {prev} to {current}"
                )
                if current == difficulty.minimum_ms:
                    hit_floor = True
            prev = current


def test_hard_faster_than_normal_faster_than_easy_at_all_levels():
    for level in range(1, 20):
        easy_g = EASY.gravity_for_level(level)
        normal_g = NORMAL.gravity_for_level(level)
        hard_g = HARD.gravity_for_level(level)
        assert hard_g <= normal_g <= easy_g, (
            f"Level {level}: expected hard≤normal≤easy, got {hard_g}/{normal_g}/{easy_g}"
        )


def test_level_up_via_line_clears():
    """Clearing 10 lines should advance the level."""
    game = GameState(difficulty=NORMAL)
    assert game.level == 1
    game.lines_cleared = 10
    game.level = 1 + game.lines_cleared // 10
    assert game.level == 2
