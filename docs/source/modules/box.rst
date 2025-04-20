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

The ``BoxDist`` distribution is constructed by transforming a base Normal distribution on \(\mathbb{R}\) to the bounded Box interval \([\text{low}, \text{high}]\) using two steps:

1. **Sigmoid Transform:**
   Maps \(z \in \mathbb{R}\) to \((0, 1)\) via the sigmoid function:
   
   .. math::
      x = \sigma(z) = \frac{1}{1 + e^{-z}}

2. **Affine Transform:**
   Maps \(x \in (0, 1)\) to \([\text{low}, \text{high}]\):
   
   .. math::
      y = \text{low} + (\text{high} - \text{low}) \cdot x

So, a sample \(z\) from the base distribution is transformed as:

.. math::
   y = \text{low} + (\text{high} - \text{low}) \cdot \sigma(z)

The probability density is adjusted using the change-of-variables formula, ensuring the resulting distribution is properly normalized over the Box bounds.

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
