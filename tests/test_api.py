import pytest
import json
import urllib.request

from tetris.api import ApiBridge, board_with_active_piece, build_snapshot, start_api_server
from tetris.difficulty import EASY, HARD, NORMAL
from tetris.game_state import GameState


def source():
    while True:
        yield "O"


def test_api_bridge_queues_valid_commands():
    bridge = ApiBridge()
    bridge.queue_command("left")
    bridge.queue_command("hard_drop")

    assert bridge.drain_commands() == ["left", "hard_drop"]
    assert bridge.drain_commands() == []


def test_api_bridge_rejects_unknown_commands():
    bridge = ApiBridge()

    with pytest.raises(ValueError):
        bridge.queue_command("teleport")


def test_snapshot_exposes_playable_state():
    game = GameState(piece_source=source(), difficulty=NORMAL)
    snapshot = build_snapshot("playing", game, NORMAL, [EASY, NORMAL, HARD])

    assert snapshot["app_state"] == "playing"
    assert snapshot["score"] == 0
    assert snapshot["difficulty"] == "Normal"
    assert snapshot["active_piece"]["name"] == "O"
    assert len(snapshot["board"]) == game.board.height
    assert len(snapshot["board"][0]) == game.board.width


def test_board_with_active_piece_overlays_current_piece():
    game = GameState(piece_source=source(), difficulty=NORMAL)
    board = board_with_active_piece(game)

    assert sum(cell == "O" for row in board for cell in row) == 4


def test_api_server_serves_state_and_accepts_commands():
    bridge = ApiBridge()
    bridge.set_snapshot({"app_state": "start"})
    server = start_api_server(bridge, port=0)
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        with urllib.request.urlopen(f"{base_url}/state") as response:
            state = json.loads(response.read().decode("utf-8"))
        assert state["app_state"] == "start"

        request = urllib.request.Request(
            f"{base_url}/command",
            data=json.dumps({"command": "start"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            accepted = json.loads(response.read().decode("utf-8"))
        assert accepted["accepted"] == "start"
        assert bridge.drain_commands() == ["start"]
    finally:
        server.shutdown()
        server.server_close()
