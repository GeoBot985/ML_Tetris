from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


API_COMMANDS = {
    "start",
    "left",
    "right",
    "soft_drop",
    "hard_drop",
    "rotate_cw",
    "rotate_ccw",
    "hold",
    "pause",
    "restart",
    "quit",
}


@dataclass
class ApiBridge:
    commands: queue.Queue[str] = field(default_factory=queue.Queue)
    _snapshot: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = snapshot

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

    def queue_command(self, command: str) -> None:
        if command not in API_COMMANDS:
            raise ValueError(f"Unknown API command: {command}")
        self.commands.put(command)

    def drain_commands(self) -> list[str]:
        commands = []
        while True:
            try:
                commands.append(self.commands.get_nowait())
            except queue.Empty:
                return commands


def build_snapshot(app_state, game_state, selected_difficulty, difficulty_order) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "app_state": app_state,
        "selected_difficulty": selected_difficulty.name,
        "difficulties": [difficulty.name for difficulty in difficulty_order],
        "available_commands": sorted(API_COMMANDS),
    }
    if game_state is None:
        return snapshot

    snapshot.update(
        {
            "score": game_state.score,
            "level": game_state.level,
            "lines_cleared": game_state.lines_cleared,
            "difficulty": game_state.difficulty.name,
            "paused": game_state.paused,
            "game_over": game_state.game_over,
            "hold_used": game_state.hold_used,
            "gravity_ms": game_state.gravity_for_level(),
            "board": board_with_active_piece(game_state),
            "locked_board": [row[:] for row in game_state.board.grid],
            "active_piece": {
                "name": game_state.active_piece.name,
                "rotation": game_state.active_piece.rotation,
                "x": game_state.active_x,
                "y": game_state.active_y,
                "cells": list(game_state.active_piece.cells()),
            },
            "next_queue": [piece.name for piece in game_state.next_queue],
            "hold_piece": game_state.hold_piece.name if game_state.hold_piece else None,
        }
    )
    return snapshot


def board_with_active_piece(game_state) -> list[list[str | None]]:
    board = [row[:] for row in game_state.board.grid]
    if game_state.active_piece is None:
        return board
    for dx, dy in game_state.active_piece.cells():
        x = game_state.active_x + dx
        y = game_state.active_y + dy
        if 0 <= y < game_state.board.height and 0 <= x < game_state.board.width:
            board[y][x] = game_state.active_piece.name
    return board


def start_api_server(bridge: ApiBridge, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._send_json({"ok": True})
            elif self.path == "/state":
                self._send_json(bridge.snapshot())
            else:
                self._send_json({"error": "Not found"}, status=404)

        def do_POST(self):
            if self.path != "/command":
                self._send_json({"error": "Not found"}, status=404)
                return
            try:
                payload = self._read_json()
                command = payload["command"]
                bridge.queue_command(command)
            except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json({"accepted": command})

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def log_message(self, format, *args):
            return

        def _read_json(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
