from __future__ import annotations


class Board:
    def __init__(self, width: int = 10, height: int = 20):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int):
        if not self.is_in_bounds(x, y):
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds")
        return self.grid[y][x]

    def set_cell(self, x: int, y: int, value) -> None:
        if not self.is_in_bounds(x, y):
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds")
        self.grid[y][x] = value

    def is_row_full(self, y: int) -> bool:
        if not (0 <= y < self.height):
            return False
        return all(cell is not None for cell in self.grid[y])

    def clear_rows(self) -> int:
        kept_rows = [row for row in self.grid if any(cell is None for cell in row)]
        cleared = self.height - len(kept_rows)
        if cleared:
            self.grid = [[None for _ in range(self.width)] for _ in range(cleared)] + kept_rows
        return cleared

    def can_place(self, cells, origin_x: int, origin_y: int) -> bool:
        for dx, dy in cells:
            x = origin_x + dx
            y = origin_y + dy
            if not self.is_in_bounds(x, y):
                return False
            if self.grid[y][x] is not None:
                return False
        return True

    def lock_piece(self, cells, origin_x: int, origin_y: int, value) -> int:
        for dx, dy in cells:
            self.set_cell(origin_x + dx, origin_y + dy, value)
        return self.clear_rows()
