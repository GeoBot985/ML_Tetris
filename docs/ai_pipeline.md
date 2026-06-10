# AI Pipeline

This project contains **two** AI stacks. The history matters, because the first one
is a cautionary tale and the second one is what actually works.

## TL;DR

| Stack | Action space | Model | Result | Status |
|---|---|---|---|---|
| Micro-action PPO (`ai_agent.training`) | one keypress/step (11 actions) | flat MLP | ~1 line | legacy / experiment |
| **Placement imitation (`ai_agent.placement`)** | **place a piece (40 slots)** | **CNN** | **~4–5 lines** | **shipping** |

Baselines on Normal: random ≈ 0 lines, heuristic coach ≈ 8–18 lines.

---

## Why the first attempt failed (micro-action PPO)

The original setup asked PPO to emit one keypress at a time (`left`, `rotate`,
`hard_drop`, …) and learn from a flat MLP over a flattened board. This failed at
three escalating levels, each verified experimentally:

1. **Reward-hackable action space.** `restart`/`pause` were in the policy's action
   set. `restart` wipes the board mid-episode with no penalty and no termination, so
   a random agent over the full action space reached game-over **0 / 5** times in
   3000 steps (vs **5 / 5** in ~62 steps for gameplay-only actions). Episodes never
   ended, so training never produced a usable signal. The shipped checkpoint's 200
   episodes all show `game_over=False` and a negative best reward.
2. **Exploration.** Even after fixing the action space, from-scratch PPO never
   stumbles onto a line clear (sparse reward), so it never sees positive reward —
   300 episodes → 0 lines.
3. **Representation.** A flat MLP cannot even *imitate* the coach: ~40% train
   accuracy. Optimal Tetris placement is spatial; a flattened board has no spatial
   structure for an MLP to exploit.

These bugs are now fixed (see CHANGELOG), but the micro-action stack remains a
documented experiment, not the recommended path.

---

## What works: placement-level imitation (`ai_agent.placement`)

### 1. Placement action space

Instead of keypresses, the agent picks **where the current piece lands**: a
`(rotation, column)` slot, `4 × 10 = 40` actions, with illegal slots masked. One
action places one piece. This shrinks the horizon ~20×, makes reward immediate, and
guarantees episodes terminate. `PlacementEnv` enumerates legal placements, executes
the chosen hard-drop, and returns the shaped reward.

### 2. CNN encoder

`CNNActorCritic` convolves over the board grid (a `1×20×10` occupancy plane) and
concatenates the active/next piece one-hots. Spatial structure is the whole game:
swapping the flat MLP for a CNN lifts coach-imitation training accuracy from ~40% to
**99%** (it can now represent the function).

### 3. Scaled imitation (DAgger / behaviour cloning)

The coach (`coach_slot`) scores every legal placement with the shaped reward and
returns the argmax — a strong, search-based expert. We behaviour-clone it. The key
finding is that **held-out accuracy scales with data**, and accuracy is what keeps
the agent alive (one bad placement compounds):

| Coach samples | Held-out accuracy | Greedy lines |
|---|---|---|
| 5,000 | 32% | 0.7 |
| 30,000 | 47% | 1.8 |
| 30,000 + dropout/weight-decay | 57% | 2.5 |
| **120,000 + regularisation** | **79%** | **~4.5** |

Regularisation (dropout `0.4`, weight decay `1e-4`) matters because the
conv→dense projection overfits badly without it. `bc_states` is the main quality
knob — more data, higher accuracy, more lines.

### 4. PPO fine-tune — and why it's off by default

`PlacementEnv` + `CNNActorCritic` include a complete, masked, GAE-based PPO
implementation (`_ppo_finetune`). **Empirically it does not improve the imitation
policy on this task**, across aggressive, conservative, and KL-anchored variants
with both the default and a survival-positive reward:

- The shaped per-placement penalty makes a *summed* return punish survival (more
  pieces placed = more accumulated penalty), so naive PPO learns to die sooner.
- Even with a survival-positive reward and a KL anchor to the BC policy, on-policy
  updates flatten the sharp, near-optimal BC action distribution faster than the
  reward can justify, and greedy play degrades (4.5 → ~1.5).

So PPO is gated behind `ppo_updates > 0` (default `0`) and, when enabled, keeps the
**best-by-eval** snapshot so it can never ship a policy worse than the BC start.
Recognising that imitation beats RL here — and *why* — is the honest result.

---

## Training

```bash
# The shipping recipe: scaled imitation of the coach (CNN, ~4–5 lines)
python train_placement_ai.py --difficulty normal

# Knobs: more data = higher accuracy = more lines
python train_placement_ai.py --difficulty normal --dagger-iterations 2   # optional DAgger refinement
```

`PlacementTrainConfig.bc_states` (default 120,000) is the main quality dial.

---

## Evaluation

```bash
python compare_ai.py --difficulty normal --episodes 10 \
    --checkpoint artifacts/ai_policy.pt \
    --placement-checkpoint artifacts/placement_policy.pt
```

Compares Random, Coach (heuristic), legacy micro-action PPO, Guarded PPO, and the
CNN placement policy on the same difficulty, and writes
`artifacts/evaluation_result.json`.
