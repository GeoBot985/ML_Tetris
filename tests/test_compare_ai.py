from __future__ import annotations

from compare_ai import RandomPolicy


def test_random_policy_is_seeded_and_reproducible():
    first = RandomPolicy(seed=123)
    second = RandomPolicy(seed=123)

    first_actions = [first.decide({}, deterministic=True) for _ in range(12)]
    second_actions = [second.decide({}, deterministic=True) for _ in range(12)]

    assert first_actions == second_actions
