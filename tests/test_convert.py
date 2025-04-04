import numpy as np
import pytest
from gymnasium.spaces import Discrete, MultiDiscrete, Box

from prob_spaces.converter import convert_to_prob_space
from prob_spaces.discrete import DiscreteDist

nvec = [2, 3, 4]
start = [0, 1, -1]


@pytest.mark.parametrize(
    "space",
    [
        Discrete(n=5, start=2),
        MultiDiscrete(nvec=nvec, start=start),
        Box(low=0, high=1, shape=(3,)),
    ],
)
def test_convert(space):

    discrete_dist = convert_to_prob_space(space)
    assert issubclass(discrete_dist.__class__, space.__class__)

    with pytest.raises(NotImplementedError):
        convert_to_prob_space((discrete_dist,))
