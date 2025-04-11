.. _discrete:

Discrete Space
==============

The ``DiscreteDist`` class extends the Gymnasium Discrete space to create categorical distributions.

Overview
--------

``DiscreteDist`` allows you to create a categorical probability distribution from a discrete action space.
It is useful for environments with a fixed number of distinct actions.

API Reference
------------

.. autoclass:: prob_spaces.discrete.DiscreteDist
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
-------------

Basic usage:

.. code-block:: python

    import torch as th
    from prob_spaces.discrete import DiscreteDist

    # Create a discrete space with 5 possible actions
    space = DiscreteDist(n=5)

    # Create logits/probabilities for each action
    probs = th.tensor([0.1, 0.2, 0.3, 0.3, 0.1])

    # Create a distribution
    dist = space(probs)

    # Sample an action
    action = dist.sample()

    # Compute log probability
    log_prob = dist.log_prob(action)

Converting from gymnasium:

.. code-block:: python

    import gymnasium as gym
    from prob_spaces.discrete import DiscreteDist

    # Create a gymnasium space
    gym_space = gym.spaces.Discrete(n=10)

    # Convert to a DiscreteDist
    space_dist = DiscreteDist.from_space(gym_space)
