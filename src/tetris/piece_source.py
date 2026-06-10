from __future__ import annotations

import random

from .pieces import all_piece_names


def seven_bag_piece_source(shuffle=None):
    if shuffle is None:
        shuffle = random.shuffle
    pieces = list(all_piece_names())
    while True:
        bag = pieces[:]
        shuffle(bag)
        yield from bag


def classic_uniform_source(seed: int | None = None):
    rng = random.Random(seed)
    pieces = list(all_piece_names())
    while True:
        yield rng.choice(pieces)
