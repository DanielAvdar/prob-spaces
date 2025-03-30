import gymnasium as gym
import pytest
import torch

from prob_spaces.box import BoxDist
from prob_spaces.discrete import DiscreteDist
from prob_spaces.multi_discrete import MultiDiscreteDist


@pytest.fixture(
    params=[
        "CartPole-v0",
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
        "Acrobot-v1",
        "phys2d/CartPole-v0",
        "phys2d/CartPole-v1",
        "phys2d/Pendulum-v0",
        "LunarLander-v3",
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3",
        "BipedalWalkerHardcore-v3",
        "CarRacing-v3",
        "Blackjack-v1",
        "FrozenLake-v1",
        "FrozenLake8x8-v1",
        "CliffWalking-v0",
        "Taxi-v3",
        "tabular/Blackjack-v0",
        # "tabular/CliffWalking-v0", todo support box int types
        # "Reacher-v2",
        "Reacher-v4",
        "Reacher-v5",
        "Pusher-v5",
        # "InvertedPendulum-v2",
        "InvertedPendulum-v4",
        "InvertedPendulum-v5",
        # "InvertedDoublePendulum-v2",
        "InvertedDoublePendulum-v4",
        "InvertedDoublePendulum-v5",
        # "HalfCheetah-v2",
        # "HalfCheetah-v3",
        "HalfCheetah-v4",
        "HalfCheetah-v5",
        # "Hopper-v2",
        # "Hopper-v3",
        "Hopper-v4",
        "Hopper-v5",
        # "Swimmer-v2",
        # "Swimmer-v3",
        "Swimmer-v4",
        "Swimmer-v5",
        # "Walker2d-v2",
        # "Walker2d-v3",
        "Walker2d-v4",
        "Walker2d-v5",
        # "Ant-v2",
        # "Ant-v3",
        "Ant-v4",
        "Ant-v5",
        # "Humanoid-v2",
        # "Humanoid-v3",
        "Humanoid-v4",
        "Humanoid-v5",
        # "HumanoidStandup-v2",
        "HumanoidStandup-v4",
        "HumanoidStandup-v5",
        # "GymV21Environment-v0",
        # "GymV26Environment-v0",
    ]
)
def fixture(request):
    env_name = request.param
    env = gym.make(env_name)
    return env


def test_gym_envs(fixture):
    env = fixture
    env.reset()
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        space_dist = MultiDiscreteDist.from_space(action_space)
        probs = torch.ones(space_dist.prob_last_dim)
        probs = torch.softmax(probs, -1)
        dist = space_dist(probs)
    elif isinstance(action_space, gym.spaces.Discrete):
        space_dist = DiscreteDist.from_space(action_space)
        probs = torch.ones(space_dist.n)
        probs = torch.softmax(probs, -1)
        dist = space_dist(probs)
    elif isinstance(action_space, gym.spaces.Box):
        space_dist = BoxDist.from_space(action_space)
        loc = torch.rand(space_dist.shape)
        scale = torch.rand(space_dist.shape)
        dist = space_dist(loc, scale)
    samples = dist.sample((100,))
    samples_np = samples.cpu().numpy()
    assert all([space_dist.contains(sample) for sample in samples_np])
    assert all([action_space.contains(sample) for sample in samples_np])
