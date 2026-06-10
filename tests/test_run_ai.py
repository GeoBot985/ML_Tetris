from run_ai import build_parser, resolve_mode


def test_run_ai_defaults_to_headless_mode():
    args = build_parser().parse_args([])

    assert resolve_mode(args) == "headless"


def test_run_ai_render_mode_is_selectable():
    args = build_parser().parse_args(["--render"])

    assert resolve_mode(args) == "render"


def test_run_ai_defaults_to_uncapped_render_speed():
    args = build_parser().parse_args(["--render"])

    assert args.fps_limit == 0
