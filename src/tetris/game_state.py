from __future__ import annotations

from collections import deque

from .board import Board
from .difficulty import NORMAL, Difficulty
from .piece_source import seven_bag_piece_source
from .pieces import Piece, make_piece
from .scoring import hard_drop_score, line_clear_score, soft_drop_score

KICK_OFFSETS = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]


class GameState:
    def __init__(self, piece_source=None, queue_size: int = 5, difficulty: Difficulty = NORMAL):
        self.piece_source = piece_source or seven_bag_piece_source()
        self.queue_size = queue_size
        self.difficulty = difficulty
        self.reset()

    def reset(self):
        self.board = Board()
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.paused = False
        self.game_over = False
        self.hold_piece = None
        self.hold_used = False
        self.next_queue = deque()
        self._fill_queue()
        self.active_piece = None
        self.active_x = 0
        self.active_y = 0
        self.spawn_piece(reset_hold=False)

    def _draw_piece_name(self):
        return next(self.piece_source)

    def _fill_queue(self):
        while len(self.next_queue) < self.queue_size:
            self.next_queue.append(make_piece(self._draw_piece_name()))

    def _spawn_position(self, piece: Piece):
        return self.board.width // 2 - 2, 0

    def spawn_piece(self, reset_hold: bool = True):
        self._fill_queue()
        self.active_piece = self.next_queue.popleft()
        self._fill_queue()
        self.active_x, self.active_y = self._spawn_position(self.active_piece)
        if reset_hold:
            self.hold_used = False
        if not self.can_active_piece_fit():
            self.game_over = True

    def can_active_piece_fit(self, piece=None, x=None, y=None):
        piece = piece or self.active_piece
        x = self.active_x if x is None else x
        y = self.active_y if y is None else y
        return self.board.can_place(piece.cells(), x, y)

    def move(self, dx: int, dy: int) -> bool:
        if self.paused or self.game_over:
            return False
        nx = self.active_x + dx
        ny = self.active_y + dy
        if self.can_active_piece_fit(x=nx, y=ny):
            self.active_x = nx
            self.active_y = ny
            return True
        return False

    def move_left(self):
        return self.move(-1, 0)

    def move_right(self):
        return self.move(1, 0)

    def soft_drop(self):
        return self._drop(award_score=True)

    def gravity_drop(self):
        return self._drop(award_score=False)

    def _drop(self, award_score: bool) -> bool:
        if self.paused or self.game_over:
            return False
        if self.move(0, 1):
            if award_score:
                self.score += soft_drop_score(1)
            return True
        self.lock_active_piece()
        return False

    def hard_drop(self):
        if self.paused or self.game_over:
            return 0
        distance = 0
        while self.move(0, 1):
            distance += 1
        self.score += hard_drop_score(distance)
        self.lock_active_piece()
        return distance

    def rotate_clockwise(self):
        return self._rotate(self.active_piece.rotate_clockwise())

    def rotate_counter_clockwise(self):
        return self._rotate(self.active_piece.rotate_counter_clockwise())

    def _rotate(self, rotated_piece: Piece):
        if self.paused or self.game_over:
            return False
        for dx, dy in KICK_OFFSETS:
            if self.can_active_piece_fit(rotated_piece, self.active_x + dx, self.active_y + dy):
                self.active_piece = rotated_piece
                self.active_x += dx
                self.active_y += dy
                return True
        return False

    def lock_active_piece(self):
        cleared = self.board.lock_piece(self.active_piece.cells(), self.active_x, self.active_y, self.active_piece.name)
        if cleared:
            self.lines_cleared += cleared
            self.score += line_clear_score(cleared, self.level)
            self.level = 1 + self.lines_cleared // 10
        self.spawn_piece()

    def hold(self):
        if self.paused or self.game_over or self.hold_used:
            return False
        self.hold_used = True
        current = self.active_piece
        if self.hold_piece is None:
            self.hold_piece = current
            self.spawn_piece(reset_hold=False)
        else:
            self.active_piece, self.hold_piece = self.hold_piece, current
            self.active_x, self.active_y = self._spawn_position(self.active_piece)
            if not self.can_active_piece_fit():
                self.game_over = True
        return True

    def toggle_pause(self):
        if not self.game_over:
            self.paused = not self.paused

    def gravity_tick(self):
        if not self.paused and not self.game_over:
            self.gravity_drop()

    def gravity_for_level(self, level: int | None = None) -> int:
        return self.difficulty.gravity_for_level(self.level if level is None else level)

    def restart(self):
        self.__init__(self.piece_source, self.queue_size, self.difficulty)
