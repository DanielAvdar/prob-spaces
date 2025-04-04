import pytest
import torch as th

from prob_spaces.discrete import DiscreteDist


@pytest.mark.parametrize(
    "n,  probs",
    [
        (22, [[0.1] * 22]),
        # (5, [[0.1, 0.1, 0.4, 0.2, 0.2], [0.1, 0.1, 0.4, 0.2, 0.2], [0.1, 0.1, 0.4, 0.2, 0.2]]),
        (5, [0.1, 0.1, 0.4, 0.2, 0.2]),
        (4, [[0.4, 0.3, 0.2, 0.1]]),
        (4, [0.4, 0.3, 0.2, 0.1]),
        (3, [[0.3, 0.4, 0.3]]),
        (3, [0.3, 0.4, 0.3]),
        (2, [[0.5, 0.5]]),
        (2, [0.5, 0.5]),
        (1, [[1.0]]),
        (1, [1.0]),
    ],
)
@pytest.mark.parametrize(
    "start",
    [
        1,
        2,
        -22,
        0,
        111,
        1000,
        10000,
        -10000,
        -1000,
        -111,
        -1,
    ],
)
def test_discrete_initialization(n, start, probs, device):
    discrete = DiscreteDist(n=n, start=start)
    prob_dist = discrete(prob=th.tensor(probs, device=device))
    sample = prob_dist.sample((100,))
    sample_list = sample.cpu().numpy().tolist()
    assert all([discrete.contains(s) for s in sample_list])
    log_probs = prob_dist.log_prob(sample)
    assert not th.isnan(log_probs).any()
    assert not th.isinf(log_probs).any()
    assert not th.isneginf(log_probs).any()
    assert th.all(log_probs <= 0)
    assert th.all(log_probs >= -1e10)
