import torch

from ai_agent.environment import TetrisEnvironment, snapshot_to_observation
from ai_agent.policy import API_ACTIONS, PPOAgentModel, PPOPolicy
from tetris.api import API_COMMANDS
from tetris.difficulty import NORMAL


def source():
    while True:
        yield "O"


def test_ppo_agent_model_produces_actor_and_critic_outputs():
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)
    observation_dim = snapshot_to_observation(env.snapshot()).shape[0]
    model = PPOAgentModel(observation_dim)

    logits, values = model(torch.zeros(observation_dim))

    assert logits.shape == (1, len(API_ACTIONS))
    assert values.shape == (1,)


def test_policy_outputs_valid_api_command():
    env = TetrisEnvironment(piece_source_factory=source, difficulty=NORMAL)
    policy = PPOPolicy.from_snapshot(env.snapshot())

    decision = policy.act_from_snapshot(env.snapshot(), deterministic=True)

    assert decision.action in API_ACTIONS
    assert decision.action in API_COMMANDS
    assert 0 <= decision.action_index < len(API_ACTIONS)
