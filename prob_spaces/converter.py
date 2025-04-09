import gymnasium as gym

from prob_spaces.box import BoxDist
from prob_spaces.dict import DictDist
from prob_spaces.discrete import DiscreteDist
from prob_spaces.multi_discrete import MultiDiscreteDist

Spaces = gym.spaces.Box | gym.spaces.Discrete | gym.spaces.MultiDiscrete
DistSpaces = BoxDist | DiscreteDist | MultiDiscreteDist | None


def convert_to_prob_space(action_space: Spaces) -> DistSpaces:
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        space_dist = MultiDiscreteDist.from_space(action_space)

    elif isinstance(action_space, gym.spaces.Discrete):
        space_dist = DiscreteDist.from_space(action_space)  # type: ignore

    elif isinstance(action_space, gym.spaces.Box):
        space_dist = BoxDist.from_space(action_space)  # type: ignore
    elif isinstance(action_space, gym.spaces.Dict):
        space_dist = DictDist()
        for k, v in action_space.spaces.items():
            space_dist[k] = convert_to_prob_space(v)
    else:
        raise NotImplementedError(f"Action space {type(action_space)} not supported")

    return space_dist
