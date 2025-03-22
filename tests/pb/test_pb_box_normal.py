import hypothesis.strategies as st
import numpy as np
import torch as th
from hypothesis import given
from hypothesis.extra import numpy as npst
from torch.distributions import Distribution, TransformedDistribution

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
    max_value = min(info.max / 2**4, 10**7)
    # max_value = info.max if width >=32 else np.finfo(np.float16).max
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
    dtype = np.dtype(f"float{64}")

    min_value = 0.00010001659393310547
    max_value = common_params["max_value"]

    scale_elements = st.floats(**common_params, min_value=min_value)
    loc_distance = st.floats(**common_params, min_value=-max_value)
    scale = draw(npst.arrays(dtype=dtype, shape=shape, elements=scale_elements))
    loc = draw(npst.arrays(dtype=dtype, shape=shape, elements=loc_distance))
    return loc, scale, low, high


@st.composite
def box_and_dist_st(draw) -> tuple[BoxDist, Distribution]:
    loc, scale, low, high = draw(normal_dist_st())
    box = BoxDist(low=low, high=high, dtype=low.dtype)
    t_loc = th.tensor(loc, requires_grad=True)
    t_scale = th.tensor(scale, requires_grad=True)
    dist_inst = box(t_loc, t_scale)
    return box, dist_inst


@given(box_and_dist=box_and_dist_st())
def test_box_contains_sample(box_and_dist):
    box, dist_inst = box_and_dist
    sample = dist_inst.sample()
    sample_np = sample.cpu().numpy().astype(box.low.dtype)
    assert box.contains(sample_np)


@given(box_and_dist=box_and_dist_st())
def test_box_contains_batch_sample(box_and_dist):
    box, dist_inst = box_and_dist
    sample = dist_inst.sample((100,))
    sample_np = sample.cpu().numpy().astype(box.low.dtype)
    contains = [box.contains(sample_np[i]) for i in range(100)]
    assert all(contains)


@given(box_and_dist=box_and_dist_st())
def test_log_prob_valid(box_and_dist):
    _box, dist_inst = box_and_dist
    sample = dist_inst.sample()
    log_prob = dist_inst.log_prob(sample)
    assert not th.any(th.isinf(log_prob))
    assert not th.any(th.isnan(log_prob))


@given(box_and_dist=box_and_dist_st())
def test_log_prob_np_valid(box_and_dist):
    box, dist_inst = box_and_dist
    sample_np = dist_inst.sample().cpu().numpy().astype(box.low.dtype)
    log_prob_np = dist_inst.log_prob(th.tensor(sample_np))
    assert not th.any(th.isinf(log_prob_np))
    assert not th.any(th.isnan(log_prob_np))


@given(box_and_dist=box_and_dist_st())
def test_batch_log_prob_valid(box_and_dist):
    _box, dist_inst = box_and_dist
    log_prob_np = dist_inst.log_prob(dist_inst.sample((100,)))
    assert not th.any(th.isinf(log_prob_np))
    assert not th.any(th.isnan(log_prob_np))


@given(box_and_dist=box_and_dist_st())
def test_dist_instance_properties(box_and_dist):
    _box, dist_inst = box_and_dist
    log_prob = dist_inst.log_prob(dist_inst.sample())
    assert isinstance(dist_inst, TransformedDistribution)
    assert log_prob.requires_grad
    assert dist_inst.rsample().requires_grad


@given(box_and_dist=box_and_dist_st())
def test_sample_shape(box_and_dist):
    box, dist_inst = box_and_dist
    sample = dist_inst.sample((100,))
    assert sample.shape == (100,) + box.low.shape
