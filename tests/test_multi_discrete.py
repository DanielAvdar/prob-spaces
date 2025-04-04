import numpy as np
import pytest
import torch as th

from prob_spaces.multi_discrete import MultiDiscreteDist


# @pytest.mark.skip
@pytest.mark.parametrize(
    "nvec, probs",
    [
        (
            [2, 3, 4],
            [
                [
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                ]
            ],
        ),
        (
            [3, 2, 4],
            [
                [
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                ]
            ],
        ),
        (
            [4, 4, 4],
            [
                [
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                ]
            ],
        ),
        (
            [4, 2, 2],
            [
                [
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                    [0.1, 0.1, 0.4, 0.2, 0.2],
                ]
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "start",
    [[0, 1, -1], [-3, 0, -0]],
)
@pytest.mark.parametrize(
    "mask",
    [
        None,
        [
            [True, False, True, False, True],
            [True, False, True, False, True],
            [True, False, True, False, True],
        ],
        [
            [False, True, False, True, False],
            [False, True, False, True, False],
            [False, True, False, True, False],
        ],
        [
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ],
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, True, False, False, False],
        ],
        [
            [True, False, False, False, False],
            [True, False, False, False, False],
            [True, False, False, False, False],
        ],
    ],
)
def test_multidiscrete_initialization(nvec, start, probs, mask, device):
    multi_discrete = MultiDiscreteDist(nvec=nvec, start=start)
    prob_dist = multi_discrete(
        prob=th.tensor(probs, dtype=th.float32, device=device, requires_grad=True),
        mask=th.tensor(mask, dtype=th.bool, device=device) if mask is not None else None,
    )
    sample = prob_dist.sample()
    sample_list = sample.cpu().numpy().tolist()
    assert len(sample_list) == len(multi_discrete.nvec)
    diffs = np.abs(multi_discrete.nvec)
    assert np.sum(diffs) == np.sum(multi_discrete.internal_mask)
    log_probs = prob_dist.log_prob(sample)
    # assert all([multi_discrete.contains(s) for s in sample_list])
    assert multi_discrete.contains(sample.cpu().numpy())
    assert not th.any(th.isinf(log_probs))
    assert not th.any(th.isneginf(log_probs))
    assert not th.any(th.isnan(log_probs))
    assert log_probs.requires_grad
