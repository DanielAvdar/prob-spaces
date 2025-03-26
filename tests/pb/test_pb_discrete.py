import hypothesis.strategies as st
import numpy as np
import torch as th
from hypothesis import given
from hypothesis.extra import numpy as npst
from scipy.special import softmax

from prob_spaces.discrete import MultiDiscreteDist


@st.composite
def start_nvec_st(draw):
    shape = draw(
        npst.array_shapes(
            min_dims=1,
            max_dims=4,
            min_side=1,
            max_side=6,
        )
    )

    width = draw(st.sampled_from([16, 32, 64]))

    dtype = np.dtype(f"int{width}")
    info = np.iinfo(dtype)

    max_value = info.max // 2
    min_value = info.min // 2
    common_params = dict(
        max_value=max_value,
    )
    elements = st.integers(**common_params, min_value=1)
    elements_distance = st.integers(**common_params, min_value=min_value)
    nvec = draw(npst.arrays(dtype=dtype, shape=shape, elements=elements))
    start = draw(npst.arrays(dtype=dtype, shape=shape, elements=elements_distance))
    nvec = np.clip(nvec, 1, a_max=max_value * 2)

    start = np.clip(start, nvec - 100, nvec - 1)
    # nvec = start + nvec
    # nvec =np.clip(nvec, 1,a_max=max_value*2)

    return start, nvec, common_params


@st.composite
def multi_d_st(draw):
    start, nvec, common_params = draw(start_nvec_st())
    prob_elements = st.floats(
        min_value=0,
        max_value=1,
    )
    max_diff = np.max(np.abs(nvec - start))
    prob_shape = (*nvec.shape, int(max_diff))
    values = draw(npst.arrays(dtype=np.float64, shape=prob_shape, elements=prob_elements))
    probs = softmax(values, 0)
    st.booleans()
    draw(
        npst.arrays(
            dtype=np.bool_,
            shape=prob_shape,
        )
    )
    torch_probs = th.tensor(probs)

    md = MultiDiscreteDist(nvec=nvec, start=start, dtype=nvec.dtype)
    dist = md(torch_probs, mask=None)
    return dist, md


@given(cat_dist=start_nvec_st())
def test_ini(cat_dist):
    start, nvec, common_params = cat_dist
    assert np.all(start < nvec)


@given(multi_d=multi_d_st())
def test_internal_mask(multi_d):
    dist, md = multi_d
    start = md.start
    nvec = md.nvec
    diffs = np.abs(start - nvec)
    th_diffs = th.tensor(diffs)
    internal_mask = th.tensor(md.internal_mask)
    assert th.sum(th_diffs).item() == th.sum(internal_mask).item()


@given(multi_d=multi_d_st())
def test_contains(multi_d):
    dist, md = multi_d
    sample = dist.sample()
    np_sample = sample.cpu().numpy()
    assert md.contains(np_sample)


@given(multi_d=multi_d_st())
def test_cat_dist(multi_d):
    dist, md = multi_d
    sample = dist.sample()
    dist.log_prob(sample)
