.. _dict:

Dict Space
==========

The ``DictDist`` class extends the Gymnasium Dict space to create nested distributions.

Overview
--------

``DictDist`` allows you to create probability distributions for complex action spaces that
consist of multiple sub-spaces organized in a dictionary structure.

API Reference
------------

.. autoclass:: prob_spaces.dict.DictDist
   :members: __call__
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic usage:

.. code-block:: python

    import torch as th
    import gymnasium as gym
    from prob_spaces.dict import DictDist
    from prob_spaces.discrete import DiscreteDist
    from prob_spaces.box import BoxDist

    # Create a dictionary space with a discrete and a box space
    space = DictDist({
        'discrete': DiscreteDist(n=3),
        'continuous': BoxDist(low=-1, high=1, shape=(2,))
    })

    # Create probabilities for each space
    probs = {
        'discrete': th.tensor([0.3, 0.4, 0.3]),
        'continuous': (th.zeros(2), th.ones(2))  # (loc, scale) for box space
    }

    # Create distributions
    dists = space(probs)

    # Sample from each distribution
    samples = {k: dist.sample() for k, dist in dists.items()}

    # Compute log probabilities
    log_probs = {k: dist.log_prob(samples[k]) for k, dist in dists.items()}
