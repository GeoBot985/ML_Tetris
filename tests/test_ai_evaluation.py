from ai_agent.evaluation import EvaluationResult, HumanBaseline, format_evaluation_report, result_to_dict


def test_evaluation_report_marks_baseline_win():
    baseline = HumanBaseline(label="Human baseline", difficulty="normal", mean_lines=40.0, mean_score=12000.0)
    result = EvaluationResult(
        episodes=3,
        max_steps=700,
        mean_lines=62.0,
        mean_score=29500.0,
        best_lines=62,
        best_score=29500,
        baseline=baseline,
    )

    report = format_evaluation_report(result)
    payload = result_to_dict(result)

    assert result.clears_baseline is True
    assert "beats the configured human baseline" in report
    assert payload["clears_baseline"] is True
    assert payload["baseline"]["difficulty"] == "normal"
