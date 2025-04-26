"""Module for converting gymnasium action spaces to probability distribution spaces."""

import gymnasium as gym

from prob_spaces.box import BoxDist
from prob_spaces.dict import DictDist
from prob_spaces.discrete import DiscreteDist
from prob_spaces.multi_discrete import MultiDiscreteDist
from prob_spaces.tuple import TupleDist

Spaces = gym.spaces.Box | gym.spaces.Discrete | gym.spaces.MultiDiscrete
DistSpaces = BoxDist | DiscreteDist | MultiDiscreteDist | None


def convert_to_prob_space(action_space: Spaces) -> DistSpaces:
    """Convert an action space into its corresponding probability distribution space.

    This function supports different types of action spaces and creates an appropriate distribution
    space for each one. Supported action spaces include `MultiDiscrete`, `Discrete`, `Box`, and
    `Dict`. For `Dict` action spaces, the function recursively converts each subspace into its
    probability distribution space.

    :param action_space: The input action space to be converted. This can be an instance of
        `gym.spaces.MultiDiscrete`, `gym.spaces.Discrete`, `gym.spaces.Box`, or `gym.spaces.Dict`.
    :type action_space: Spaces
    :raises NotImplementedError: If the input action space type is not supported.
    :return: The corresponding probability distribution space created based on the input action space
        type.

    Returns
    -------
    DistSpaces
        The corresponding probability distribution space created based on the input action space type.

    """
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
    elif isinstance(action_space, gym.spaces.Tuple):
        space_list = []
        for v in action_space.spaces:
            space_list.append(convert_to_prob_space(v))
        space_dist = TupleDist(space_list)
    else:
        raise NotImplementedError(f"Action space {type(action_space)} not supported")

    return space_dist
