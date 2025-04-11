MultiDiscrete Space
==================

The ``MultiDiscreteDist`` class extends the Gymnasium MultiDiscrete space to create categorical distributions
for multiple discrete variables.

Overview
--------

``MultiDiscreteDist`` allows you to create probability distributions for action spaces with multiple
discrete components, each with its own cardinality.

API Reference
------------

.. autoclass:: prob_spaces.multi_discrete.MultiDiscreteDist
   :members:
   :undoc-members:
   :show-inheritance:

Key Attributes
-------------

* ``nvec``: Array of integers representing the number of values for each discrete variable
* ``start``: Optional starting indices for each variable
* ``internal_mask``: Automatically generated mask to ensure valid actions

Usage Examples
-------------

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
