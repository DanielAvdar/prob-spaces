.. _tuple:

Tuple Space
===========

The ``TupleDist`` class extends the Gymnasium Tuple space to handle combinations of multiple sub-spaces, generating composite probability distributions.

Overview
--------

``TupleDist`` is useful when working with action spaces that require different sub-spaces organized in a tuple structure. It generates distributions corresponding to each sub-space, which can be either discrete or continuous.

API Reference
-------------

.. autoclass:: prob_spaces.tuple.TupleDist
   :members: __call__
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic usage with discrete and continuous spaces:

.. code-block:: python

    import torch as th
    from prob_spaces.tuple import TupleDist
    from prob_spaces.discrete import DiscreteDist
    from prob_spaces.box import BoxDist

    # Create a tuple space consisting of discrete and continuous components
    space = TupleDist(spaces=(
        DiscreteDist(n=3),
        BoxDist(low=-1, high=1, shape=(2,))
    ))

    # Define probabilities
    probs = (
        th.tensor([0.1, 0.6, 0.3]),                     # Probabilities for discrete component
        (th.tensor([0.0, 0.0]), th.tensor([0.5, 0.5]))  # Mean and variance for continuous component
    )

    # Create the tuple of distributions
    distributions = space(prob=probs)

    # Sample from the distribution tuple
    samples = [dist.sample() for dist in distributions]

With masking for discrete spaces:

.. code-block:: python

    import torch as th
    from prob_spaces.tuple import TupleDist
    from prob_spaces.discrete import DiscreteDist
    from prob_spaces.box import BoxDist

    space = TupleDist(spaces=(
        DiscreteDist(n=4),
        BoxDist(low=-2, high=2, shape=(3,))
    ))

    probs = (
        th.tensor([0.25, 0.25, 0.25, 0.25]),                    # Probabilities for discrete component
        (th.tensor([0.0, 0.0, 0.0]), th.tensor([1.0, 1.0, 1.0])) # Mean, variance for continuous component
    )

    # Apply mask to discrete component to disallow specific actions
    masks = (
        th.tensor([1, 0, 1, 1], dtype=th.bool),
        None  # No mask for the continuous component
    )

    distributions = space(prob=probs, mask=masks)

    # Sampling respecting masks
    samples = [dist.sample() for dist in distributions]
