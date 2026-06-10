# AI Evaluation Report

- Difficulty: normal
- Episodes per policy: 5
- Max steps: 300
- Seed: 1

| Policy | Mean lines | Mean score | Game-over % | Safety corrections |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.0 | 129 | 100% | n/a |
| Coach (heuristic) | 17.4 | 3850 | 40% | n/a |
| PPO | 0.2 | 316 | 100% | n/a |
| Guarded PPO | 0.2 | 318 | 100% | 1.9% |
| CNN Placement (DAgger+PPO) | 4.2 | 420 | 100% | n/a |
