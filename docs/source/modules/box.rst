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
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
------------

* Support for arbitrary continuous ranges with lower and upper bounds
* Built-in transformation to enforce bounds using sigmoid and affine transforms
* Customizable base distribution (defaults to Normal)

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
