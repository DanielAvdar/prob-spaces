.. _discrete:

Discrete Space
==============

The ``DiscreteDist`` class extends the Gymnasium Discrete space to create categorical distributions.

Overview
--------

``DiscreteDist`` allows you to create a categorical probability distribution from a discrete action space.
It is useful for environments with a fixed number of distinct actions.

API Reference
-------------

.. autoclass:: prob_spaces.discrete.DiscreteDist
   :members: __call__
   :undoc-members:
   :show-inheritance:

Detailed Descriptions
---------------------

The `DiscreteDist` class includes the following key methods:

### __init__

```python
def __init__(self, n: int, start: int = 0):
    """
    Initialize DiscreteDist with the number of categories and an optional start offset.

    Parameters
    ----------
    n : int
        The number of categories in the discrete space.
    start : int, optional
        The starting offset for the categories. Defaults to 0.

    Returns
    -------
    None
    """
```

### __call__

```python
def __call__(self, prob: th.Tensor, mask: th.Tensor = None) -> MaskedCategorical:
    """
    Compute and return a masked categorical distribution.

    Compute a masked categorical distribution based on the given probability tensor and an
    optional mask. The distribution incorporates specific probabilities and constraints defined
    by the provided input.

    Parameters
    ----------
    prob : th.Tensor
        A tensor representing the probabilities for each category.
    mask : th.Tensor, optional
        A tensor specifying a mask to limit the valid categories. Defaults to a tensor
        of ones if not provided.

    Returns
    -------
    MaskedCategorical
        A MaskedCategorical distribution constructed with given probabilities, mask, and
        starting values.

    Raises
    ------
    ValueError
        If the `prob` tensor does not match the expected shape.
    """
```

### from_space

```python
@classmethod
def from_space(cls, space: spaces.Discrete) -> "DiscreteDist":
    """
    Create a DiscreteDist from a gymnasium Discrete space.

    Parameters
    ----------
    space : spaces.Discrete
        The gymnasium Discrete space to convert.

    Returns
    -------
    DiscreteDist
        An instance of DiscreteDist created from the given gymnasium Discrete space.

    Raises
    ------
    TypeError
        If the input space is not a valid gymnasium Discrete space.
    """
```

Usage Examples
--------------

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
