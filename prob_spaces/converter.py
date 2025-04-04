import gymnasium as gym

from prob_spaces.box import BoxDist
from prob_spaces.discrete import DiscreteDist
from prob_spaces.multi_discrete import MultiDiscreteDist

Spaces = gym.spaces.Box | gym.spaces.Discrete | gym.spaces.MultiDiscrete
DistSpaces = BoxDist | DiscreteDist | MultiDiscreteDist


def convert_to_prob_space(action_space: Spaces) -> DistSpaces:
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        space_dist = MultiDiscreteDist.from_space(action_space)

    elif isinstance(action_space, gym.spaces.Discrete):
        space_dist = DiscreteDist.from_space(action_space)  # type: ignore

    elif isinstance(action_space, gym.spaces.Box):
        space_dist = BoxDist.from_space(action_space)  # type: ignore
    else:
        raise TypeError(f"Unsupported action space type: {type(action_space)}")

    return space_dist
