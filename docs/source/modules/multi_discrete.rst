.. _multi_discrete:

MultiDiscrete Space
===================

The ``MultiDiscreteDist`` class extends the Gymnasium MultiDiscrete space to create categorical distributions
for multiple discrete variables.

Overview
--------

``MultiDiscreteDist`` allows you to create probability distributions for action spaces with multiple
discrete components, each with its own cardinality.

API Reference
------------

.. autoclass:: prob_spaces.multi_discrete.MultiDiscreteDist
   :members: __call__
   :undoc-members:
   :show-inheritance:

Key Attributes
--------------

* ``nvec``: Array of integers representing the number of values for each discrete variable
* ``start``: Optional starting indices for each variable
* ``internal_mask``: Automatically generated mask to ensure valid actions

Detailed Descriptions
---------------------

The `MultiDiscreteDist` class includes the following key methods:

### __init__

```python
def __init__(
    self,
    nvec: NDArray[np.integer[Any]] | list[int],
    dtype: str | type[np.integer[Any]] = np.int64,
    seed: int | np.random.Generator | None = None,
    start: NDArray[np.integer[Any]] | list[int] | None = None,
):
    """
    Initialize MultiDiscreteDist with nvec, dtype, seed, and start.

    Parameters
    ----------
    nvec : NDArray[np.integer[Any]] | list[int]
        Array of integers representing the number of values for each discrete variable.
    dtype : str | type[np.integer[Any]], optional
        The data type of the MultiDiscrete space. Defaults to np.int64.
    seed : int | np.random.Generator | None, optional
        The seed for random number generation. Defaults to None.
    start : NDArray[np.integer[Any]] | list[int] | None, optional
        Optional starting indices for each variable. Defaults to None.

    Returns
    -------
    None
    """
```

### __call__

```python
def __call__(self, prob: th.Tensor, mask: th.Tensor = None) -> MaskedCategorical:
    """
    Apply a transformation to the input probability tensor and optional mask.

    Create a `MaskedCategorical` distribution by reshaping the input probabilities, applying an
    optional mask, and combining with an internal mask.

    Parameters
    ----------
    prob : th.Tensor
        A tensor containing probabilities to be reshaped and used in constructing the
        distribution.
    mask : th.Tensor, optional
        An optional boolean tensor for masking specific probabilities before creating
        the distribution. Defaults to None.

    Returns
    -------
    MaskedCategorical
        A `MaskedCategorical` distribution object created with reshaped probabilities and
        combined masking information.

    Raises
    ------
    ValueError
        If the `prob` tensor does not match the expected shape.
    """
```

### from_space

```python
@classmethod
def from_space(cls, space: spaces.MultiDiscrete) -> "MultiDiscreteDist":
    """
    Create a MultiDiscreteDist from a gymnasium MultiDiscrete space.

    Parameters
    ----------
    space : spaces.MultiDiscrete
        The gymnasium MultiDiscrete space to convert.

    Returns
    -------
    MultiDiscreteDist
        An instance of MultiDiscreteDist created from the given gymnasium MultiDiscrete space.

    Raises
    ------
    TypeError
        If the input space is not a valid gymnasium MultiDiscrete space.
    """
```

Usage Examples
--------------

Basic usage:

.. code-block:: python

    import numpy as np
    import torch as th
    from prob_spaces.multi_discrete import MultiDiscreteDist

    # Create a multi-discrete space with 3 variables:
    # - First variable has 2 possible values (0, 1)
    # - Second variable has 3 possible values (0, 1, 2)
    # - Third variable has 4 possible values (0, 1, 2, 3)
    nvec = np.array([2, 3, 4])
    space = MultiDiscreteDist(nvec=nvec)

    # Create logits for each variable
    # The shape should be (nvec shape) + (max(nvec) + 1)
    # In this case: (3, 5)
    probs = th.ones((3, 5))

    # Create a distribution
    dist = space(probs)

    # Sample an action
    action = dist.sample()

    # Compute log probability
    log_prob = dist.log_prob(action)

With masking:

.. code-block:: python

    import numpy as np
    import torch as th
    from prob_spaces.multi_discrete import MultiDiscreteDist

    nvec = np.array([2, 3, 4])
    space = MultiDiscreteDist(nvec=nvec)

    probs = th.ones((3, 5))

    # Create a mask to disallow certain actions
    mask = th.ones((3, 5), dtype=th.bool)
    mask[0, 1] = False  # Disallow action 1 for first variable

    # Create a distribution with the mask
    dist = space(probs, mask=mask)
