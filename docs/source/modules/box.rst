.. _box:

Box Space
=========

The ``BoxDist`` class extends the Gymnasium Box space to create continuous probability distributions.

Overview
--------

``BoxDist`` allows you to create continuous probability distributions for action spaces with bounded
values. By default, it uses a transformed Normal distribution to ensure samples are within the
specified bounds.

API Reference
-------------

.. autoclass:: prob_spaces.box.BoxDist
   :members: __call__
   :undoc-members:
   :show-inheritance:

Key Features
------------

* Support for arbitrary continuous ranges with lower and upper bounds
* Built-in transformation to enforce bounds using sigmoid and affine transforms
* Customizable base distribution (defaults to Normal)

Mathematical Details
-------------------

The ``BoxDist`` distribution is constructed by transforming a base Normal distribution on :math:`\mathbb{R}` to the bounded Box interval :math:`[\mathrm{low},\ \mathrm{high}]` using three steps:

1. **Normal Distribution (Base):**
   The base distribution is a Normal (Gaussian) distribution on :math:`\mathbb{R}`. The user must provide the parameters:
   - ``loc``: the mean of the Normal distribution
   - ``scale``: the standard deviation of the Normal distribution

   .. note::
      Here, :math:`z` is sampled from :math:`\mathcal{N}(\text{loc},\ \text{scale})`, i.e., :math:`z \sim \mathcal{N}(\text{loc},\ \text{scale})`.

2. **Sigmoid Transform:**
   Maps :math:`z \in \mathbb{R}` to :math:`(0, 1)` via the sigmoid function:

   .. math::
      x = \sigma(z) = \frac{1}{1 + e^{-z}}

3. **Affine Transform:**
   Maps :math:`x \in (0, 1)` to :math:`[\mathrm{low},\ \mathrm{high}]`:

   .. math::
      y = \mathrm{low} + (\mathrm{high} - \mathrm{low}) \cdot x

So, a sample :math:`z` from the base distribution is transformed as:

.. math::
   y = \mathrm{low} + (\mathrm{high} - \mathrm{low}) \cdot \sigma(z)

The probability density is adjusted using the change-of-variables formula, ensuring the resulting distribution is properly normalized over the Box bounds.

Detailed Descriptions
---------------------

The `BoxDist` class includes the following key methods:

### __init__

```python
def __init__(
    self,
    low: SupportsFloat | NDArray[Any],
    high: SupportsFloat | NDArray[Any],
    shape: Sequence[int] | None = None,
    dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
    seed: int | np.random.Generator | None = None,
    dist: None | Type[th.distributions.Distribution] = None,
):
    """
    Initialize BoxDist with bounds, shape, dtype, and base distribution.

    Parameters
    ----------
    low : SupportsFloat | NDArray[Any]
        The lower bound of the Box space.
    high : SupportsFloat | NDArray[Any]
        The upper bound of the Box space.
    shape : Sequence[int] | None, optional
        The shape of the Box space. If None, it will be inferred from `low` and `high`.
    dtype : type[np.floating[Any]] | type[np.integer[Any]], optional
        The data type of the Box space. Defaults to np.float32.
    seed : int | np.random.Generator | None, optional
        The seed for random number generation. Defaults to None.
    dist : None | Type[th.distributions.Distribution], optional
        The base distribution to use. Defaults to None, which uses Normal distribution.

    Returns
    -------
    None
    """
```

### __call__

```python
def __call__(self, loc: th.Tensor, scale: th.Tensor) -> th.distributions.Distribution:
    """
    Generate a transformed probability distribution.

    Construct a base distribution, apply a sequence of transformations, and return the resulting
    transformed distribution.

    Parameters
    ----------
    loc : th.Tensor
        A tensor specifying the location parameters for the base distribution.
    scale : th.Tensor
        A tensor specifying the scale parameters for the base distribution.

    Returns
    -------
    th.distributions.Distribution
        A transformed distribution object derived from the specified base distribution and
        transformations.

    Raises
    ------
    ValueError
        If the `loc` and `scale` tensors do not match the expected shape.
    """
```

### from_space

```python
@classmethod
def from_space(cls, space: spaces.Box) -> "BoxDist":
    """
    Create a BoxDist from a gymnasium Box space.

    Parameters
    ----------
    space : spaces.Box
        The gymnasium Box space to convert.

    Returns
    -------
    BoxDist
        An instance of BoxDist created from the given gymnasium Box space.

    Raises
    ------
    TypeError
        If the input space is not a valid gymnasium Box space.
    """
```

Usage Examples
--------------

Basic usage:

.. code-block:: python

    import numpy as np
    import torch as th
    from prob_spaces.box import BoxDist

    # Create a 2D box space with values between -1 and 1
    space = BoxDist(low=-1, high=1, shape=(2,))

    # Create parameters for the distribution
    loc = th.zeros(2)    # Mean
    scale = th.ones(2)   # Standard deviation

    # Create a distribution
    dist = space(loc, scale)

    # Sample an action
    action = dist.sample()

    # Compute log probability
    log_prob = dist.log_prob(action)

Converting from gymnasium:

.. code-block:: python

    import gymnasium as gym
    from prob_spaces.box import BoxDist

    # Create a gymnasium space
    gym_space = gym.spaces.Box(low=-1, high=1, shape=(3,))

    # Convert to a BoxDist
    space_dist = BoxDist.from_space(gym_space)
