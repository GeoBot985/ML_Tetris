# SafetyWrapper

`ai_agent.safety.SafetyWrapper` wraps a `PPOPolicy` and ensures that every executed action is legal and not obviously catastrophic.

## Decision flow

```
snapshot
    │
    └─ predict_from_snapshot()          ← single model inference
           │
           ├─ is_legal_action()?
           │       ├─ yes → action tentatively accepted
           │       └─ no  → correction_reason = "illegal_action"
           │                action = coach_action(snapshot)
           │
           ├─ risk_score() >= threshold?
           │       ├─ no  → action accepted
           │       └─ yes → correction_reason = "high_risk"
           │                action = coach_action(snapshot)
           │
           └─ is_legal_action(corrected_action)?
                   ├─ yes → done
                   └─ no  → correction_reason = "safety_fallback"
                            action = _first_legal_action(snapshot)
                            (priority: hard_drop → rotate → left/right → soft_drop → hold)
```

The policy is called **exactly once** per decision regardless of correction path.

## Legality checking

`is_legal_action(snapshot, action)` rebuilds the board from the snapshot and tests the action:

| Action | Test |
|---|---|
| `left` | `board.can_place(cells, x−1, y)` |
| `right` | `board.can_place(cells, x+1, y)` |
| `soft_drop` | `board.can_place(cells, x, y+1)` |
| `hard_drop` | `board.can_place(cells, x, y)` (always legal if piece exists) |
| `rotate_cw/ccw` | Tests all 6 SRS kick offsets — legal if any offset fits |
| `hold` | Legal only if `hold_used == False` |
| `pause/restart/quit` | Always illegal during gameplay |

App-state guards:
- In `"start"` state, only `"start"` is legal.
- In `"game_over"` state, only `"restart"` is legal.

## Risk scoring

```python
risk = 0.6 × (stack_height / board_height)
     + 0.4 × hole_density
     + value_penalty        # up to 0.25 if critic value < 0
```

`risk_threshold` defaults to `0.78`. The wrapper is intentionally conservative: it would rather over-correct than execute a move that buries the stack.

## SafetyDecision

Every call returns a `SafetyDecision` dataclass:

| Field | Type | Description |
|---|---|---|
| `model_decision` | `PolicyDecision` | Raw policy output (logits, value, entropy) |
| `executed_action` | `str` | Action that was actually applied |
| `corrected` | `bool` | Whether the model's action was overridden |
| `correction_reason` | `str \| None` | `"illegal_action"`, `"high_risk"`, `"safety_fallback"`, or `None` |
| `risk_score` | `float` | Board risk at the time of decision |
| `legal_action` | `bool` | Whether the model's original action was legal |

## Usage

```python
from ai_agent.safety import SafetyWrapper
from ai_agent.training import load_policy_from_checkpoint
from tetris.difficulty import NORMAL

policy = load_policy_from_checkpoint("artifacts/ai_policy.pt")
wrapper = SafetyWrapper(policy, difficulty=NORMAL, risk_threshold=0.78)

decision = wrapper.decide(snapshot)
print(decision.executed_action, decision.corrected, decision.correction_reason)
```

## Tuning

- **`risk_threshold`**: Lower values make the wrapper more aggressive about correcting (more coach actions, fewer model actions). 0.78 is a reasonable default for a partially-trained policy.
- The coach correction (`_correction_action`) calls `coach_action()` which is deterministic given the snapshot — the same board state always produces the same fallback.
