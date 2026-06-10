# AI Evaluation Example

Generated from `compare_ai.py --checkpoint artifacts/ai_policy.pt --episodes 1 --max-steps 200 --difficulty normal`.

## Summary

- Difficulty: normal
- Episodes per policy: 1
- Max steps: 200
- Seed: 1

| Policy | Mean lines | Mean score | Game-over % | Safety corrections |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.0 | 144 | 100% | n/a |
| Coach (heuristic) | 17.0 | 3596 | 0% | n/a |
| PPO | 1.0 | 464 | 100% | n/a |
| Guarded PPO | 1.0 | 466 | 100% | 1.6% |

## Notes

- This is a short smoke comparison, not a full benchmark sweep.
- `Guarded PPO` includes safety correction accounting from the wrapper layer.
- Re-run the command above after training to refresh the numbers.
