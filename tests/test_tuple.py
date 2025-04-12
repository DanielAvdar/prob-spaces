import pytest
import torch
from gymnasium.spaces import Box, Discrete, Tuple

from prob_spaces import convert_to_prob_space
from prob_spaces.tuple import TupleDist


@pytest.mark.parametrize(
    "spaces, probs, masks",
    [
        (
            (Discrete(3), Box(-1.0, 1.0, (2,))),
            (torch.tensor([0.1, 0.6, 0.3]), (torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.8]))),
            (None, None),
        ),
        (
            (Discrete(4), Box(-2.0, 2.0, (3,))),
            (torch.tensor([0.25, 0.25, 0.25, 0.25]), (torch.tensor([0.3, 0.5, 0.2]), torch.tensor([0.4, 0.4, 0.2]))),
            (torch.tensor([1, 0, 1, 1], dtype=torch.bool), None),
        ),
    ],
)
def test_tupledist_initialization_and_call(spaces, probs, masks):
    tuple_space = Tuple(spaces=spaces)

    tuple_dist = convert_to_prob_space(tuple_space)
    assert isinstance(tuple_dist, TupleDist)
    assert tuple_dist.spaces == spaces

    dist_result = tuple_dist(prob=probs, mask=masks)
    assert isinstance(dist_result, tuple)
    assert len(dist_result) == len(spaces)


def test_tupledist_with_invalid_probabilities():
    spaces = (Discrete(3), Box(-1.0, 1.0, (2,)))
    tuple_dist = TupleDist(spaces=spaces)

    invalid_probs = ("invalid", (torch.tensor([0.2, 0.8]), torch.tensor([0.5, 0.5])))
    with pytest.raises(TypeError):
        tuple_dist(prob=invalid_probs)


def test_tupledist_with_missing_probabilities():
    spaces = (Discrete(3),)
    tuple_dist = TupleDist(spaces=spaces)

    missing_probs = ()
    with pytest.raises(IndexError):
        tuple_dist(prob=missing_probs)
