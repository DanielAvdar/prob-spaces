import pytest
import torch as th

from prob_spaces.discrete import DiscreteDist, MultiDiscreteDist


@pytest.mark.parametrize(
    "n,  probs",
    [
        (5, [[0.1, 0.1, 0.4, 0.2, 0.2], [0.1, 0.1, 0.4, 0.2, 0.2], [0.1, 0.1, 0.4, 0.2, 0.2]]),
        (5, [[0.1, 0.1, 0.4, 0.2, 0.2], [0.1, 0.1, 0.4, 0.2, 0.2], [0.1, 0.1, 0.4, 0.2, 0.2]]),
        (4, [[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]]),
        (3, [[0.3, 0.4, 0.3]]),
    ],
)
@pytest.mark.parametrize(
    "start",
    [1, 2, -22, 0, 111],
)
def test_discrete_initialization(n, start, probs):
    discrete = DiscreteDist(n=n, start=start)
    prob_dist = discrete(prob=th.tensor(probs))
    sample = prob_dist.sample()
    sample_list = sample.cpu().numpy().tolist()
    assert all([discrete.contains(s) for s in sample_list])
    prob_dist.log_prob(sample)


@pytest.mark.skip
@pytest.mark.parametrize(
    "nvec, probs",
    [
        (
            [2, 3, 4],
            [[[0.1, 0.1, 0.4, 0.2, 0.2], [0.1, 0.1, 0.4, 0.2, 0.2], [0.1, 0.1, 0.4, 0.2, 0.2]]] * 3,
        ),  # Example with incorrect probs size
        # ([2, 3, 4], [[0.5, 0.5], [0.3, 0.3, 0.4], [0.2, 0.2, 0.3, 0.3]]),  # Correct probs size
        # ([3, 2,6], [[0.3, 0.3, 0.4], [0.6, 0.4]]),  # Another correct probs size
    ],
)
@pytest.mark.parametrize(
    "start",
    [[0, 1, -1], [0, 0, -0]],
)
def test_multidiscrete_initialization(nvec, start, probs):
    multi_discrete = MultiDiscreteDist(nvec=nvec, start=start)
    prob_dist = multi_discrete(prob=th.tensor(probs, dtype=th.float32))
    sample = prob_dist.sample()
    sample_list = sample.cpu().numpy().tolist()
    assert len(sample_list) == len(nvec)
    prob_dist.log_prob(sample)
