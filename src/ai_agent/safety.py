from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tetris.board import Board
from tetris.game_state import KICK_OFFSETS
from tetris.pieces import make_piece

from .policy import PolicyDecision, PPOPolicy
from .rewards import profile_board
from .training import coach_action


GAMEPLAY_ACTIONS = {"left", "right", "soft_drop", "hard_drop", "rotate_cw", "rotate_ccw", "hold"}
CONTROL_ACTIONS = {"start", "restart", "pause", "quit"}


@dataclass(frozen=True)
class SafetyDecision:
    model_decision: PolicyDecision
    executed_action: str
    corrected: bool
    correction_reason: str | None
    risk_score: float
    legal_action: bool


class SafetyWrapper:
    def __init__(
        self,
        policy: PPOPolicy,
        *,
        difficulty: Any,
        risk_threshold: float = 0.78,
    ):
        self.policy = policy
        self.difficulty = difficulty
        self.risk_threshold = risk_threshold

    def decide(self, snapshot: dict[str, Any], deterministic: bool = True) -> SafetyDecision:
        model_decision = self.policy.predict_from_snapshot(snapshot, deterministic=deterministic)
        action = model_decision.action
        legal_action = self.is_legal_action(snapshot, action)
        risk_score = self.risk_score(snapshot, model_decision)
        corrected = False
        correction_reason = None

        if not legal_action:
            action = self._correction_action(snapshot)
            corrected = True
            correction_reason = "illegal_action"
        elif risk_score >= self.risk_threshold:
            action = self._correction_action(snapshot)
            corrected = True
            correction_reason = "high_risk"

        if not self.is_legal_action(snapshot, action):
            action = self._first_legal_action(snapshot)
            corrected = True
            correction_reason = correction_reason or "safety_fallback"

        return SafetyDecision(
            model_decision=model_decision,
            executed_action=action,
            corrected=corrected,
            correction_reason=correction_reason,
            risk_score=risk_score,
            legal_action=legal_action,
        )

    def risk_score(self, snapshot: dict[str, Any], model_decision: PolicyDecision) -> float:
        profile = profile_board(snapshot)
        height_ratio = profile.stack_height / profile.height if profile.height else 0.0
        hole_ratio = profile.hole_density
        value_penalty = 0.0
        value = float(model_decision.value.detach().cpu().item())
        if value < 0:
            value_penalty = min(0.25, abs(value) / 10.0)
        return min(1.0, 0.6 * height_ratio + 0.4 * hole_ratio + value_penalty)

    def is_legal_action(self, snapshot: dict[str, Any], action: str) -> bool:
        app_state = snapshot.get("app_state", "playing")
        if app_state == "start":
            return action == "start"
        if app_state == "game_over":
            return action == "restart"
        if action in CONTROL_ACTIONS:
            return False
        if action not in GAMEPLAY_ACTIONS:
            return False

        if action == "hold":
            return not snapshot.get("hold_used", False)

        if action in {"pause", "restart", "start", "quit"}:
            return False

        return self._is_move_legal(snapshot, action)

    def _is_move_legal(self, snapshot: dict[str, Any], action: str) -> bool:
        board = self._board_from_snapshot(snapshot)
        active_piece = snapshot.get("active_piece") or {}
        piece_name = active_piece.get("name")
        if piece_name is None:
            return False
        piece = make_piece(piece_name, int(active_piece.get("rotation", 0)))
        x = int(active_piece.get("x", 0))
        y = int(active_piece.get("y", 0))

        if action == "left":
            return board.can_place(piece.cells(), x - 1, y)
        if action == "right":
            return board.can_place(piece.cells(), x + 1, y)
        if action == "soft_drop":
            return board.can_place(piece.cells(), x, y + 1)
        if action == "hard_drop":
            return board.can_place(piece.cells(), x, y)
        if action in {"rotate_cw", "rotate_ccw"}:
            rotated = piece.rotate_clockwise() if action == "rotate_cw" else piece.rotate_counter_clockwise()
            for dx, dy in KICK_OFFSETS:
                if board.can_place(rotated.cells(), x + dx, y + dy):
                    return True
            return False
        return False

    def _board_from_snapshot(self, snapshot: dict[str, Any]) -> Board:
        rows = snapshot.get("locked_board") or snapshot.get("board") or []
        width = len(rows[0]) if rows else 10
        height = len(rows)
        board = Board(width, height)
        board.grid = [list(row) for row in rows]
        return board

    def _correction_action(self, snapshot: dict[str, Any]) -> str:
        return coach_action(snapshot, difficulty=self.difficulty)[0]

    def _first_legal_action(self, snapshot: dict[str, Any]) -> str:
        candidates = ["hard_drop", "rotate_cw", "rotate_ccw", "left", "right", "soft_drop", "hold"]
        for action in candidates:
            if self.is_legal_action(snapshot, action):
                return action
        return "hard_drop"
