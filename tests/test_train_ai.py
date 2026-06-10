from train_ai import build_parser


def test_train_ai_defaults_parse():
    args = build_parser().parse_args([])

    assert args.episodes == 200
    assert args.difficulty == "normal"
    assert str(args.checkpoint).endswith("artifacts\\ai_policy.pt") or str(args.checkpoint).endswith("artifacts/ai_policy.pt")
