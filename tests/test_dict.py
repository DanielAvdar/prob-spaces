import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete

from prob_spaces import convert_to_prob_space
from prob_spaces.dict import DictDist


@pytest.mark.parametrize(
    "spaces, probs, masks",
    [
        (
            {"box_space": Box(-1.0, 1.0, (2,)), "discrete_space": Discrete(3)},
            {
                "box_space": (torch.tensor([0.2, 0.8]), torch.tensor([0.5, 0.5])),
                "discrete_space": torch.tensor([0.1, 0.4, 0.5]),
            },
            None,
        ),
        (
            {"box_space": Box(-1.0, 1.0, (3,)), "discrete_space": Discrete(4)},
            {
                "box_space": (torch.tensor([0.3, 0.6, 0.1]), torch.tensor([0.2, 0.5, 0.3])),
                "discrete_space": torch.tensor([0.25, 0.25, 0.25, 0.25]),
            },
            {"discrete_space": torch.tensor([1, 0, 1, 1], dtype=torch.bool)},
        ),
    ],
)
def test_dictdist_initialization_and_call(spaces, probs, masks):
    dict_space = Dict(spaces=spaces)

    dict_dist = convert_to_prob_space(dict_space)
    assert isinstance(dict_dist, DictDist)
    assert dict_dist.spaces == spaces

    dist_result = dict_dist(prob=probs, mask=masks)
    assert isinstance(dist_result, dict)
    assert set(dist_result.keys()) == set(spaces.keys())


def test_dictdist_with_invalid_probabilities():
    spaces = {"box_space": Box(-1.0, 1.0, (2,)), "discrete_space": Discrete(3)}
    dict_dist = DictDist(spaces=spaces)

    invalid_probs = {"box_space": "invalid", "discrete_space": np.array([0.1, 0.5, 0.4])}
    with pytest.raises(TypeError):
        dict_dist(prob=invalid_probs)


def test_dictdist_with_missing_probabilities():
    spaces = {"discrete_space": Discrete(3)}
    dict_dist = DictDist(spaces=spaces)

    missing_probs = {}
    with pytest.raises(KeyError):
        dict_dist(prob=missing_probs)
