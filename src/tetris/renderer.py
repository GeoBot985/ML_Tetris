from __future__ import annotations

import pygame
from pygame import gfxdraw

from .difficulty import EASY, HARD, NORMAL
from .pieces import PIECE_SHAPES


PIECE_COLORS = {
    "I": (0, 240, 240),
    "O": (240, 240, 0),
    "T": (160, 0, 240),
    "S": (0, 240, 0),
    "Z": (240, 0, 0),
    "J": (0, 0, 240),
    "L": (240, 160, 0),
}

START_SCREEN_DIFFICULTIES = (EASY, NORMAL, HARD)

GAMEPLAY_CONTROLS = [
    ("Left / Right", "Move"),
    ("Down", "Soft drop"),
    ("Space", "Hard drop"),
    ("Z / Up", "Rotate CW"),
    ("X", "Rotate CCW"),
    ("C", "Hold"),
    ("P", "Pause"),
    ("R", "Restart"),
    ("Esc", "Quit"),
]

GAMEPLAY_CONTROLS = [
    ("← / →", "Move"),
    ("↓", "Soft drop"),
    ("Space", "Hard drop"),
    ("Z / ↑", "Rotate CW"),
    ("X", "Rotate CCW"),
    ("C", "Hold"),
    ("P", "Pause"),
    ("R", "Restart"),
    ("Esc", "Quit"),
]

GAMEPLAY_CONTROLS = [
    ("Left / Right", "Move"),
    ("Down", "Soft drop"),
    ("Space", "Hard drop"),
    ("Z / Up", "Rotate CW"),
    ("X", "Rotate CCW"),
    ("C", "Hold"),
    ("P", "Pause"),
    ("R", "Restart"),
    ("Esc", "Quit"),
]


class Renderer:
    def __init__(
        self,
        board_width: int = 10,
        board_height: int = 20,
        cell_size: int = 30,
        sidebar_width: int = 260,
        *,
        vector_mode: bool = False,
    ):
        self.board_width = board_width
        self.board_height = board_height
        self.cell_size = cell_size
        self.margin = 20
        self.sidebar_width = sidebar_width
        self.vector_mode = vector_mode
        self.surface_width = self.margin * 3 + board_width * cell_size + sidebar_width
        self.surface_height = self.margin * 2 + board_height * cell_size
        self.sidebar_x = self.margin * 2 + board_width * cell_size
        if not pygame.font.get_init():
            pygame.font.init()
        self.fonts = {
            "title": pygame.font.SysFont("arial", 44, bold=True),
            "overlay_title": pygame.font.SysFont("arial", 32, bold=True),
            "section": pygame.font.SysFont("arial", 24, bold=True),
            "body": pygame.font.SysFont("arial", 24),
            "sidebar": pygame.font.SysFont("arial", 22),
            "overlay": pygame.font.SysFont("arial", 20),
            "small": pygame.font.SysFont("arial", 18),
            "start_small": pygame.font.SysFont("arial", 20),
        }
        self.overlay_surface = pygame.Surface((self.surface_width, self.surface_height), pygame.SRCALPHA)
        self.overlay_surface.fill((0, 0, 0, 120))

    def board_to_screen(self, x: int, y: int):
        return self.margin + x * self.cell_size, self.margin + y * self.cell_size

    def draw(self, surface, game_state, diagnostics=None, high_score: int = 0):
        surface.fill((18, 18, 24))
        if self.vector_mode:
            self._draw_vector_backdrop(surface)
        self._draw_board(surface, game_state)
        self._draw_active_piece(surface, game_state)
        self._draw_sidebar(surface, game_state, diagnostics=diagnostics, high_score=high_score)
        if game_state.paused:
            self._draw_overlay(surface, "PAUSED", "Press P to resume")
        if game_state.game_over:
            self._draw_game_over_overlay(surface, game_state, high_score=high_score)

    def draw_start_screen(
        self,
        surface,
        selected_difficulty,
        selected_action="play",
        status_message: str | None = None,
        checkpoint_ready: bool = True,
        job_active: bool = False,
        training_progress=None,
        human_hint_count: int = 0,
        high_score: int = 0,
    ):
        surface.fill((18, 18, 24))
        if self.vector_mode:
            self._draw_vector_backdrop(surface, start_screen=True)
        self._draw_start_screen_overlay(
            surface,
            selected_difficulty,
            selected_action=selected_action,
            status_message=status_message,
            checkpoint_ready=checkpoint_ready,
            job_active=job_active,
            training_progress=training_progress,
            human_hint_count=human_hint_count,
            high_score=high_score,
        )

    # ------------------------------------------------------------------ drawing

    def _draw_board(self, surface, game_state):
        board_rect = pygame.Rect(self.margin, self.margin, self.board_width * self.cell_size, self.board_height * self.cell_size)
        pygame.draw.rect(surface, (20, 20, 28), board_rect, border_radius=14)
        pygame.draw.rect(surface, (58, 58, 78), board_rect, width=2, border_radius=14)
        for y, row in enumerate(game_state.board.grid):
            for x, cell in enumerate(row):
                if cell is not None:
                    self._draw_cell(surface, x, y, PIECE_COLORS[cell], filled=True)
        for x in range(self.board_width):
            for y in range(self.board_height):
                rect = pygame.Rect(*self.board_to_screen(x, y), self.cell_size, self.cell_size)
                if self.vector_mode:
                    pygame.draw.rect(surface, (42, 42, 52), rect, 1, border_radius=4)
                else:
                    pygame.draw.rect(surface, (40, 40, 48), rect, 1)

    def _draw_active_piece(self, surface, game_state):
        if game_state.active_piece is None:
            return
        for dx, dy in game_state.active_piece.cells():
            self._draw_cell(surface, game_state.active_x + dx, game_state.active_y + dy, PIECE_COLORS[game_state.active_piece.name], filled=True)

    def _draw_cell(self, surface, x, y, color, *, filled: bool = False):
        rect = pygame.Rect(*self.board_to_screen(x, y), self.cell_size, self.cell_size)
        if self.vector_mode:
            self._draw_vector_tile(surface, rect, color, filled=filled)
        else:
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (0, 0, 0), rect, 1)

    def _draw_mini_piece(self, surface, piece_name: str, center_x: int, top_y: int, mini_size: int = 8):
        """Draw a small piece shape centered at (center_x, top_y)."""
        cells = PIECE_SHAPES[piece_name][0]
        color = PIECE_COLORS[piece_name]
        xs = [dx for dx, _ in cells]
        ys = [dy for _, dy in cells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max_x - min_x + 1
        span_y = max_y - min_y + 1
        offset_x = center_x - (span_x * mini_size) // 2
        offset_y = top_y
        for dx, dy in cells:
            rx = offset_x + (dx - min_x) * mini_size
            ry = offset_y + (dy - min_y) * mini_size
            rect = pygame.Rect(rx, ry, mini_size - 1, mini_size - 1)
            pygame.draw.rect(surface, color, rect)

    def _draw_sidebar(self, surface, game_state, diagnostics=None, high_score: int = 0):
        panel = pygame.Rect(self.sidebar_x, self.margin, self.sidebar_width - self.margin, self.surface_height - self.margin * 2)
        self._draw_panel(surface, panel)

        accent = (240, 240, 240)
        muted = (185, 185, 195)

        x = self.sidebar_x + 18
        y = self.margin + 16
        for label, value in [
            ("SCORE", str(game_state.score)),
            ("BEST", f"{max(0, int(high_score)):,}"),
            ("LEVEL", str(game_state.level)),
            ("LINES", str(game_state.lines_cleared)),
        ]:
            surface.blit(self.fonts["section"].render(label, True, muted), (x, y))
            surface.blit(self.fonts["sidebar"].render(value, True, accent), (x, y + 26))
            y += 54

        # Hold piece
        surface.blit(self.fonts["section"].render("HOLD", True, muted), (x, y))
        if game_state.hold_piece is not None:
            hold_color = accent if not game_state.hold_used else (120, 120, 130)
            self._draw_mini_piece(surface, game_state.hold_piece.name, x + 48, y + 28, mini_size=9)
        else:
            surface.blit(self.fonts["small"].render("—", True, muted), (x, y + 28))
        y += 58

        # Next queue — draw shapes
        surface.blit(self.fonts["section"].render("NEXT", True, muted), (x, y))
        y += 28
        for piece in list(game_state.next_queue)[:3]:
            self._draw_mini_piece(surface, piece.name, x + 44, y, mini_size=9)
            y += 26

        # Controls hint at the bottom
        hint_lines = [
            "← → Move",
            "↓ Soft  Space Hard",
            "Z/↑ Rotate  X CCW",
            "C Hold  P Pause",
            "R Restart  Esc Quit",
        ]
        hint_y = self.surface_height - 18 - len(hint_lines) * 20
        for index, line in enumerate(hint_lines):
            surface.blit(self.fonts["small"].render(line, True, muted), (x, hint_y + index * 20))

        if diagnostics is not None:
            diag_top = hint_y - 112
            self._draw_diagnostics(surface, diagnostics, x, diag_top, panel.width - 36)

    def _draw_overlay(self, surface, title, subtitle):
        surface.blit(self.overlay_surface, (0, 0))
        box = pygame.Rect(0, 0, 360, 120)
        box.center = (self.surface_width // 2, self.surface_height // 2)
        self._draw_panel(surface, box, fill=(20, 20, 28), outline=(90, 90, 110))
        self._blit_centered(surface, self.fonts["overlay_title"].render(title, True, (240, 240, 240)), 0, box.top + 20)
        self._blit_centered(surface, self.fonts["overlay"].render(subtitle, True, (190, 190, 200)), 0, box.top + 64)

    def _draw_game_over_overlay(self, surface, game_state, high_score: int = 0):
        surface.blit(self.overlay_surface, (0, 0))
        box = pygame.Rect(0, 0, 380, 200)
        box.center = (self.surface_width // 2, self.surface_height // 2)
        self._draw_panel(surface, box, fill=(20, 20, 28), outline=(90, 90, 110))

        accent = (240, 240, 240)
        muted = (190, 190, 200)
        highlight = (120, 220, 255)

        self._blit_centered(surface, self.fonts["overlay_title"].render("GAME OVER", True, highlight), 0, box.top + 14)

        stats = [
            f"Score   {game_state.score:,}",
            f"Best    {max(0, int(high_score)):,}",
            f"Lines   {game_state.lines_cleared}",
            f"Level   {game_state.level}",
        ]
        for i, line in enumerate(stats):
            self._blit_centered(surface, self.fonts["overlay"].render(line, True, accent), 0, box.top + 54 + i * 27)

        self._blit_centered(surface, self.fonts["small"].render("Press R to restart", True, muted), 0, box.top + 166)

    def _draw_start_screen_overlay(
        self,
        surface,
        selected_difficulty,
        selected_action="play",
        status_message: str | None = None,
        checkpoint_ready: bool = True,
        job_active: bool = False,
        training_progress=None,
        human_hint_count: int = 0,
        high_score: int = 0,
    ):
        accent = (240, 240, 240)
        muted = (180, 180, 190)
        soft = (205, 205, 220)
        highlight = (120, 220, 255)

        self._blit_centered(surface, self.fonts["title"].render("TETRIS RL", True, accent), 0, 20)
        self._blit_centered(surface, self.fonts["body"].render("Reinforcement-Learning Workbench", True, soft), 0, 72)

        # Three columns: mode | difficulty | controls
        col_w = (self.surface_width - self.margin * 4) // 3
        col1_x = self.margin
        col2_x = col1_x + col_w + self.margin
        col3_x = col2_x + col_w + self.margin
        box_top = 108
        box_h = 310

        action_box = pygame.Rect(col1_x, box_top, col_w, box_h)
        difficulty_box = pygame.Rect(col2_x, box_top, col_w, 200)
        controls_box = pygame.Rect(col3_x, box_top, col_w, box_h)

        self._draw_panel(surface, action_box, fill=(24, 24, 34), outline=(70, 70, 92))
        self._draw_panel(surface, difficulty_box, fill=(24, 24, 34), outline=(70, 70, 92))
        self._draw_panel(surface, controls_box, fill=(24, 24, 34), outline=(70, 70, 92))

        # Mode column
        menu_entries = [
            ("play", "Play", "Human-controlled game"),
            ("train", "Train ML", "Headless reinforcement learning"),
            ("watch", "Watch ML", "Agent plays visually"),
            ("evaluate", "Evaluate ML", "Score against baseline"),
            ("benchmark", "Benchmark ML", "Measure inference latency"),
            ("clear_ml", "Reset AI", "Delete checkpoint & logs"),
            ("quit", "Quit", "Exit"),
        ]
        surface.blit(self.fonts["section"].render("Mode", True, soft), (action_box.left + 14, action_box.top + 10))
        for index, (key, label, description) in enumerate(menu_entries):
            is_selected = key == selected_action
            y = action_box.top + 42 + index * 36
            marker = "▶ " if is_selected else "  "
            color = highlight if is_selected else muted
            surface.blit(self.fonts["body"].render(f"{marker}{label}", True, color), (action_box.left + 14, y))
            surface.blit(self.fonts["small"].render(description, True, (150, 150, 165)), (action_box.left + 16, y + 19))

        # Difficulty column
        surface.blit(self.fonts["section"].render("Difficulty", True, soft), (difficulty_box.left + 14, difficulty_box.top + 10))
        for index, diff in enumerate(START_SCREEN_DIFFICULTIES):
            is_selected = diff.name == selected_difficulty.name
            prefix = "▶ " if is_selected else "  "
            color = highlight if is_selected else muted
            surface.blit(self.fonts["body"].render(f"{prefix}{diff.name}", True, color), (difficulty_box.left + 14, difficulty_box.top + 50 + index * 42))
            gravity_label = f"  {diff.start_ms}ms → {diff.minimum_ms}ms"
            surface.blit(self.fonts["small"].render(gravity_label, True, (150, 150, 165)), (difficulty_box.left + 14, difficulty_box.top + 72 + index * 42))

        # Controls column
        surface.blit(self.fonts["section"].render("Controls", True, soft), (controls_box.left + 14, controls_box.top + 10))
        for i, (key, action) in enumerate(GAMEPLAY_CONTROLS):
            cy = controls_box.top + 46 + i * 28
            surface.blit(self.fonts["small"].render(key, True, highlight), (controls_box.left + 14, cy))
            surface.blit(self.fonts["small"].render(action, True, muted), (controls_box.left + 90, cy))

        # Training progress box under difficulty
        training_box = pygame.Rect(difficulty_box.left, difficulty_box.bottom + 10, col_w, 86)
        self._draw_panel(surface, training_box, fill=(22, 22, 32), outline=(82, 82, 104))
        surface.blit(self.fonts["small"].render("Training", True, soft), (training_box.left + 12, training_box.top + 8))

        if training_progress is None:
            summary_lines = ["untrained", "ready to start"]
            source_label = "source: none"
        else:
            summary_lines = [
                f"{training_progress.episodes} eps  {training_progress.total_steps} steps",
                f"best {training_progress.best_lines} lines  {training_progress.best_reward:.1f} reward",
            ]
            source_label = "checkpoint" if training_progress.source == "checkpoint" else "live log"

        for i, line in enumerate(summary_lines):
            surface.blit(self.fonts["small"].render(line, True, muted if i else accent), (training_box.left + 12, training_box.top + 28 + i * 18))
        surface.blit(self.fonts["small"].render(source_label, True, muted), (training_box.right - 90, training_box.top + 8))
        surface.blit(self.fonts["small"].render(f"hints {human_hint_count}", True, muted), (training_box.left + 12, training_box.bottom - 24))
        surface.blit(self.fonts["small"].render(f"high score {max(0, int(high_score)):,}", True, muted), (training_box.left + 12, training_box.bottom - 44))

        # Checkpoint status
        checkpoint_label = "Checkpoint: ready ✓" if checkpoint_ready else "Checkpoint: missing"
        checkpoint_color = (120, 220, 120) if checkpoint_ready else (255, 160, 160)
        surface.blit(self.fonts["small"].render(checkpoint_label, True, checkpoint_color), (col2_x, self.surface_height - 56))

        # Bottom nav
        nav_lines = [
            "Left/Right — cycle mode    Up/Down — difficulty    Enter — launch",
        ]
        if job_active:
            nav_lines.append("S — stop active job")
        for i, line in enumerate(nav_lines):
            self._blit_centered(surface, self.fonts["small"].render(line, True, muted), 0, self.surface_height - 36 + i * 20)

        if status_message:
            status_lines = status_message.splitlines() or [status_message]
            status_height = max(34, 12 + len(status_lines) * 22)
            status_box = pygame.Rect(self.margin, self.surface_height - 90 - status_height, self.surface_width - self.margin * 2, status_height)
            self._draw_panel(surface, status_box, fill=(26, 26, 40), outline=(90, 90, 110))
            for i, line in enumerate(status_lines):
                surface.blit(self.fonts["small"].render(line, True, accent), (status_box.left + 12, status_box.top + 8 + i * 22))

    def _draw_start_screen_overlay(
        self,
        surface,
        selected_difficulty,
        selected_action="play",
        status_message: str | None = None,
        checkpoint_ready: bool = True,
        job_active: bool = False,
        training_progress=None,
        human_hint_count: int = 0,
        high_score: int = 0,
    ):
        accent = (240, 240, 240)
        muted = (180, 180, 190)
        soft = (205, 205, 220)
        highlight = (120, 220, 255)

        self._blit_centered(surface, self.fonts["title"].render("TETRIS RL", True, accent), 0, 18)
        self._blit_centered(surface, self.fonts["body"].render("Reinforcement-Learning Workbench", True, soft), 0, 70)

        left_x = self.margin
        left_w = 230
        middle_x = left_x + left_w + 16
        middle_w = 150
        right_x = middle_x + middle_w + 16
        right_w = self.surface_width - right_x - self.margin
        box_top = 108

        action_box = pygame.Rect(left_x, box_top, left_w, 340)
        difficulty_box = pygame.Rect(middle_x, box_top, middle_w, 198)
        training_box = pygame.Rect(middle_x, difficulty_box.bottom + 10, middle_w, 132)
        controls_box = pygame.Rect(right_x, box_top, right_w, 340)

        self._draw_panel(surface, action_box, fill=(24, 24, 34), outline=(70, 70, 92))
        self._draw_panel(surface, difficulty_box, fill=(24, 24, 34), outline=(70, 70, 92))
        self._draw_panel(surface, training_box, fill=(22, 22, 32), outline=(82, 82, 104))
        self._draw_panel(surface, controls_box, fill=(24, 24, 34), outline=(70, 70, 92))

        menu_entries = [
            ("play", "Play", "Human-controlled game"),
            ("train", "Train ML", "Headless reinforcement learning"),
            ("watch", "Watch ML", "Agent plays visually"),
            ("clear_ml", "Reset AI", "Delete checkpoint and logs"),
            ("evaluate", "Evaluate ML", "Score against baseline"),
            ("benchmark", "Benchmark ML", "Measure inference latency"),
            ("quit", "Quit", "Exit"),
        ]
        surface.blit(self.fonts["section"].render("Mode", True, soft), (action_box.left + 14, action_box.top + 10))
        for index, (key, label, description) in enumerate(menu_entries):
            is_selected = key == selected_action
            y = action_box.top + 36 + index * 40
            marker = "> " if is_selected else "  "
            color = highlight if is_selected else muted
            surface.blit(self.fonts["body"].render(f"{marker}{label}", True, color), (action_box.left + 14, y))
            surface.blit(self.fonts["small"].render(description, True, (150, 150, 165)), (action_box.left + 16, y + 18))

        surface.blit(self.fonts["section"].render("Difficulty", True, soft), (difficulty_box.left + 14, difficulty_box.top + 10))
        for index, diff in enumerate(START_SCREEN_DIFFICULTIES):
            is_selected = diff.name == selected_difficulty.name
            prefix = "> " if is_selected else "  "
            color = highlight if is_selected else muted
            row_y = difficulty_box.top + 50 + index * 48
            surface.blit(self.fonts["body"].render(f"{prefix}{diff.name}", True, color), (difficulty_box.left + 14, row_y))
            gravity_label = f"{diff.start_ms}ms -> {diff.minimum_ms}ms"
            surface.blit(self.fonts["small"].render(gravity_label, True, (150, 150, 165)), (difficulty_box.left + 14, row_y + 22))

        surface.blit(self.fonts["small"].render("Training", True, soft), (training_box.left + 12, training_box.top + 8))
        if training_progress is None:
            summary_lines = ["untrained", "ready to start"]
            source_label = "source: none"
        else:
            summary_lines = [
                f"{training_progress.episodes} eps  {training_progress.total_steps} steps",
                f"best {training_progress.best_lines} lines  {training_progress.best_reward:.1f} reward",
            ]
            source_label = "checkpoint" if training_progress.source == "checkpoint" else "live log"

        for i, line in enumerate(summary_lines):
            surface.blit(self.fonts["small"].render(line, True, muted if i else accent), (training_box.left + 12, training_box.top + 28 + i * 20))
        surface.blit(self.fonts["small"].render(source_label, True, muted), (training_box.left + 12, training_box.bottom - 24))
        surface.blit(self.fonts["small"].render(f"hints {human_hint_count}", True, muted), (training_box.left + 12, training_box.bottom - 66))
        surface.blit(self.fonts["small"].render(f"high score {max(0, int(high_score)):,}", True, muted), (training_box.left + 12, training_box.bottom - 46))

        surface.blit(self.fonts["section"].render("Controls", True, soft), (controls_box.left + 14, controls_box.top + 10))
        for i, (key, action) in enumerate(GAMEPLAY_CONTROLS):
            cy = controls_box.top + 46 + i * 28
            surface.blit(self.fonts["small"].render(key, True, highlight), (controls_box.left + 14, cy))
            surface.blit(self.fonts["small"].render(action, True, muted), (controls_box.left + 90, cy))

        checkpoint_label = "Checkpoint: ready \u2713" if checkpoint_ready else "Checkpoint: missing"
        checkpoint_color = (120, 220, 120) if checkpoint_ready else (255, 160, 160)
        surface.blit(self.fonts["small"].render(checkpoint_label, True, checkpoint_color), (middle_x, self.surface_height - 60))

        nav_lines = ["Left/Right = cycle mode    Up/Down = difficulty    Enter = launch"]
        if job_active:
            nav_lines.append("S = stop active job")
        for i, line in enumerate(nav_lines):
            self._blit_centered(surface, self.fonts["small"].render(line, True, muted), 0, self.surface_height - 36 + i * 20)

        if status_message:
            status_lines = status_message.splitlines() or [status_message]
            status_height = max(34, 12 + len(status_lines) * 22)
            status_box = pygame.Rect(self.margin, self.surface_height - 90 - status_height, self.surface_width - self.margin * 2, status_height)
            self._draw_panel(surface, status_box, fill=(26, 26, 40), outline=(90, 90, 110))
            for i, line in enumerate(status_lines):
                surface.blit(self.fonts["small"].render(line, True, accent), (status_box.left + 12, status_box.top + 8 + i * 22))

    def _draw_diagnostics(self, surface, diagnostics, x, top, width):
        panel = pygame.Rect(x - 4, top, width, 100)
        self._draw_panel(surface, panel, fill=(22, 22, 30), outline=(90, 90, 110))

        accent = (240, 240, 240)
        muted = (170, 170, 185)
        value = diagnostics.value if diagnostics.value == diagnostics.value else 0.0
        lines = [
            ("MODEL", f"{diagnostics.model_action} → {diagnostics.executed_action}"),
            ("VALUE", f"{value:.3f}"),
            ("ENTROPY", f"{diagnostics.entropy:.3f}"),
            ("RISK", f"{diagnostics.risk_score:.2f}"),
            ("TOP", ", ".join(f"{name}:{score:.2f}" for name, score in diagnostics.top_actions)),
        ]
        if diagnostics.correction_reason is not None:
            lines.insert(1, ("CORRECTED", diagnostics.correction_reason))

        cursor_y = top + 8
        for index, (label, text) in enumerate(lines):
            surface.blit(self.fonts["small"].render(label, True, muted), (x, cursor_y + index * 23))
            surface.blit(self.fonts["small"].render(text, True, accent), (x + 68, cursor_y + index * 23))

    def _blit_centered(self, surface, rendered_text, y_offset, top, align_left=False):
        rect = rendered_text.get_rect()
        rect.top = top + y_offset
        if align_left:
            rect.left = self.margin
        else:
            rect.centerx = self.surface_width // 2
        surface.blit(rendered_text, rect)

    def _draw_panel(self, surface, rect, *, fill=(28, 28, 38), outline=(60, 60, 75)):
        if self.vector_mode:
            pygame.draw.rect(surface, fill, rect, border_radius=12)
            pygame.draw.rect(surface, outline, rect, width=2, border_radius=12)
            accent_rect = pygame.Rect(rect.left + 8, rect.top + 8, max(0, rect.width - 16), 4)
            pygame.draw.rect(surface, (255, 255, 255), accent_rect, border_radius=2)
            if accent_rect.width > 0:
                overlay = pygame.Surface((accent_rect.width, accent_rect.height), pygame.SRCALPHA)
                overlay.fill((255, 255, 255, 28))
                surface.blit(overlay, accent_rect.topleft)
        else:
            pygame.draw.rect(surface, fill, rect, border_radius=12)
            pygame.draw.rect(surface, outline, rect, width=2, border_radius=12)

    def _draw_vector_backdrop(self, surface, *, start_screen: bool = False):
        width, height = surface.get_size()
        palette = [(32, 36, 52, 40), (54, 72, 110, 28), (120, 220, 255, 18)]
        offsets = [0, 160, 320]
        for index, color in enumerate(palette):
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            for y in range(-height, height * 2, 60):
                pygame.draw.line(overlay, color, (0, y + offsets[index]), (width, y + 120 + offsets[index]), 2)
            surface.blit(overlay, (0, 0))
        if start_screen:
            glow = pygame.Surface((width, height), pygame.SRCALPHA)
            pygame.draw.circle(glow, (80, 180, 255, 36), (width // 2, 100), 220)
            surface.blit(glow, (0, 0))

    def _draw_vector_tile(self, surface, rect, color, *, filled: bool):
        radius = max(4, self.cell_size // 7)
        shadow = rect.move(3, 3)
        pygame.draw.rect(surface, (0, 0, 0), shadow, border_radius=radius)
        gfxdraw.box(surface, rect, color)
        pygame.draw.rect(surface, color, rect, border_radius=radius)
        pygame.draw.rect(surface, (0, 0, 0), rect, width=1, border_radius=radius)
        if filled:
            highlight = pygame.Rect(rect.left + 3, rect.top + 3, max(1, rect.width - 6), max(1, rect.height // 4))
            highlight_surface = pygame.Surface((highlight.width, highlight.height), pygame.SRCALPHA)
            highlight_surface.fill((255, 255, 255, 40))
            surface.blit(highlight_surface, highlight.topleft)
        if self.vector_mode:
            cx = rect.centerx
            cy = rect.centery
            points = [(rect.left, cy), (cx, rect.top), (rect.right, cy), (cx, rect.bottom)]
            pygame.draw.polygon(surface, (255, 255, 255, 20), points, width=0)
