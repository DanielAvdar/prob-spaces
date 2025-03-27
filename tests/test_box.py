import numpy as np
import pytest
import torch as th
from gymnasium.spaces import Box

from prob_spaces.box import BoxDist


@pytest.mark.parametrize(
    "low, high, shape",
    [
        (0.0, 1.0, (3,)),
        (-1.0, 1.0, (2, 2)),
        (np.array([-1.0, -2.0]), np.array([1.0, 2.0]), (2,)),
    ],
)
def test_boxdist_initialization(low, high, shape):
    box_dist = BoxDist(low=low, high=high, shape=shape)
    assert isinstance(box_dist, Box)
    assert box_dist.shape == shape


@pytest.mark.parametrize(
    "low, high, shape,",
    [
        (
            0.0,
            1.0,
            (3,),
        ),
        (
            -1.0,
            1.0,
            (2, 2),
        ),
    ],
)
def test_boxdist_initialization_with_dist(
    low,
    high,
    shape,
):
    box_dist = BoxDist(
        low=low,
        high=high,
        shape=shape,
    )
    assert isinstance(box_dist, Box)
    assert box_dist.shape == shape
    # assert box_dist.dist_class == dist/


@pytest.mark.parametrize(
    "low, high, shape",
    [
        (0.0, 1.0, (3,)),
        (0.0, 1.0, (1,)),
        (-1.1, 1.0, (1,)),
        (-1.1, 1.0, (1, 2)),
        (-1.1, 1.0, (1, 2, 2)),
        (0, 1.0, (1,)),
        (-0.1, 1.0, (3,)),
        (44.0, 77.0, (3,)),
        (-55.0, 1.0, (3, 2)),
        (-55.0, 1.0, (3, 2, 1)),
        (1.0, 2.0, (3,)),
    ],
)
def test_boxdist_call(
    low,
    high,
    shape,
    device,
):
    box_dist = BoxDist(low=low, high=high, shape=shape)
    loc = th.zeros(shape, dtype=th.float32, device=device)
    scale = th.ones(shape, device=device)
    dist_instance = box_dist(
        loc.requires_grad_(True).clone(),
        scale.requires_grad_(True).clone(),
    )
    sample = dist_instance.sample()
    sample_np = sample.cpu().numpy()
    assert box_dist.contains(sample_np)
    log_prob = dist_instance.log_prob(sample)
    assert not th.any(th.isinf(log_prob))
    assert log_prob.requires_grad
    assert dist_instance.rsample().requires_grad


@pytest.mark.parametrize(
    "low, high",
    [
        (
            np.zeros((1,), dtype=np.float32),
            np.ones((1,), dtype=np.float32),
        ),
        (
            np.zeros((1, 2), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
        ),
        (
            -np.ones((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        ),
        (
            -np.ones((1, 2), dtype=np.float32) * 3,
            np.ones((1, 2), dtype=np.float32) * 77,
        ),
        (
            np.array([-1.0, -2.0, 0, 1], dtype=np.float32),
            np.array([1.0, 2.0, 1, 2], dtype=np.float32),
        ),
    ],
)
@pytest.mark.parametrize(
    "dist",
    [
        None,
        th.distributions.Normal,
        th.distributions.Beta,
    ],
)
def test_boxdist_np(low, high, dist):
    shape = low.shape
    box_dist = BoxDist(low=low, high=high, shape=shape, dist=dist)
    loc = th.zeros(shape, dtype=th.float32) + 0.001
    scale = th.ones(shape)
    if dist == th.distributions.Beta:
        loc += 1

    dist_instance = box_dist(
        loc.requires_grad_(True).clone(),
        scale.requires_grad_(True).clone(),
    )
    sample = dist_instance.sample()
    sample_np = sample.cpu().numpy()
    assert box_dist.contains(sample_np)
    log_prob = dist_instance.log_prob(sample)
    assert not th.any(th.isinf(log_prob))
    assert log_prob.requires_grad
    assert dist_instance.rsample().requires_grad
