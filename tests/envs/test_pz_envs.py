import pytest
import torch
from pettingzoo.classic import chess_v6, connect_four_v3, go_v5, tictactoe_v3

from prob_spaces.converter import convert_to_prob_space
from prob_spaces.discrete import DiscreteDist


@pytest.fixture(
    params=[
        chess_v6.env(),
        connect_four_v3.env(),
        go_v5.env(),
        tictactoe_v3.env(),
        # rps_v2.env(),
    ]
)
def fixture(request):
    env = request.param
    # env = gym.make(env_name)
    return env


def test_gym_envs(fixture):
    env = fixture
    env.reset()
    for step in range(100):
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
                env.step(action)
                continue

            mask = observation["action_mask"]
            th_mask = torch.tensor(mask, dtype=torch.bool)
            action_space = env.action_space(agent)
            space_dist = convert_to_prob_space(action_space)
            assert isinstance(space_dist, DiscreteDist)
            probs = torch.ones(space_dist.n)
            probs = torch.softmax(probs, -1)
            dist = space_dist(probs, mask=th_mask)

            samples = dist.sample((100,))
            samples_np = samples.cpu().numpy()
            assert all([space_dist.contains(sample) for sample in samples_np]), f"for {agent} at step {step}"
            assert all([action_space.contains(sample) for sample in samples_np]), f"for {agent} at step {step}"
            assert torch.all(th_mask[samples] != 0), f"for {agent} at step {step}, mask is not satisfied"
            action = samples_np[0]
            env.step(action)
            # break
