import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import torch as th
from hypothesis import given
from hypothesis.extra import numpy as npst
from torch.distributions import TransformedDistribution

from prob_spaces.box import BoxDist


@st.composite
def low_high_st(draw):
    shape = draw(
        npst.array_shapes(
            min_dims=1,
            max_dims=5,
            min_side=1,
            max_side=9,
        )
    )

    width = draw(st.sampled_from([16, 32, 64]))
    # width = draw(st.sampled_from([32, 64]))
    # width = draw(st.sampled_from([ 64]))
    dtype = np.dtype(f"float{width}")

    info = np.finfo(dtype)
    max_value = min(info.max / 2**4, 10**6)
    common_params = dict(
        allow_nan=False,
        allow_infinity=False,
        max_value=max_value,
        exclude_min=True,
        exclude_max=True,
        width=width,
    )
    elements = st.floats(**common_params, min_value=-100)
    elements_distance = st.floats(**common_params, min_value=1)
    low = draw(npst.arrays(dtype=dtype, shape=shape, elements=elements))
    high = draw(npst.arrays(dtype=dtype, shape=shape, elements=elements_distance))
    high = low + high

    return low, high, common_params


@st.composite
def normal_dist_st(draw):
    low, high, common_params = draw(low_high_st())
    shape = low.shape
    common_params["width"]
    dtype = np.dtype(f"float{64}")

    min_value = 0.00010001659393310547
    max_value = common_params["max_value"]

    scale_elements = st.floats(**common_params, min_value=min_value)
    loc_distance = st.floats(**common_params, min_value=-max_value)
    scale = draw(npst.arrays(dtype=dtype, shape=shape, elements=scale_elements))
    loc = draw(npst.arrays(dtype=dtype, shape=shape, elements=loc_distance))
    return loc, scale, low, high


@st.composite
def beta_dist_st(draw):
    low, high, common_params = draw(low_high_st())
    shape = low.shape
    min_value = 0.00010001659393310547
    dtype = np.dtype(f"float{64}")
    concentration1_elements = st.floats(
        **common_params,
        min_value=min_value,
    )
    concentration2_elements = st.floats(
        **common_params,
        min_value=min_value,
    )
    concentration1 = draw(npst.arrays(dtype=dtype, shape=shape, elements=concentration1_elements))
    concentration2 = draw(npst.arrays(dtype=dtype, shape=shape, elements=concentration2_elements))
    return concentration1, concentration2, low, high


@given(normal_dist=normal_dist_st())
@hp.settings(
    deadline=None,
)
def test_box_normal_distribution(normal_dist):
    loc, scale, low, high = normal_dist
    box = BoxDist(low=low, high=high, dtype=low.dtype)
    t_loc = th.tensor(loc, requires_grad=True)
    t_scale = th.tensor(scale, requires_grad=True)
    dist_inst = box(t_loc, t_scale)
    sample = dist_inst.sample()
    sample_np = sample.cpu().numpy().astype(low.dtype)
    assert box.contains(sample_np)
    log_prob = dist_inst.log_prob(sample)
    assert not th.any(th.isinf(log_prob))

    assert isinstance(dist_inst, TransformedDistribution)
    assert log_prob.requires_grad
    assert dist_inst.rsample().requires_grad


@given(beta_dist=beta_dist_st())
@hp.settings(
    deadline=None,
)
def test_box_beta_distribution(beta_dist):
    concentration1, concentration2, low, high = beta_dist
    box = BoxDist(low=low, high=high, dtype=low.dtype)
    t_concentration1 = th.tensor(concentration1, requires_grad=True)
    t_concentration2 = th.tensor(concentration2, requires_grad=True)
    dist_inst = box(t_concentration1, t_concentration2)
    sample = dist_inst.sample()
    sample_np = sample.cpu().numpy().astype(low.dtype)
    assert box.contains(sample_np)
    log_prob = dist_inst.log_prob(sample)
    assert not th.any(th.isinf(log_prob))

    assert isinstance(dist_inst, TransformedDistribution)
    assert log_prob.requires_grad
    assert dist_inst.rsample().requires_grad
