Space Converter
===============

The ``converter`` module provides utilities to convert Gymnasium spaces to prob-spaces equivalents.

Overview
--------

The main function in this module is ``convert_to_prob_space``, which automatically converts
any supported Gymnasium space to its corresponding probability space.

API Reference
-------------

.. autofunction:: prob_spaces.converter.convert_to_prob_space

Usage Examples
--------------

Converting a simple space:

.. code-block:: python

    import gymnasium as gym
    from prob_spaces.converter import convert_to_prob_space

    # Create a Gymnasium space
    gym_space = gym.spaces.Discrete(5)

    # Convert to a probability space
    prob_space = convert_to_prob_space(gym_space)

Converting complex spaces:

.. code-block:: python

    import gymnasium as gym
    import numpy as np
    from prob_spaces.converter import convert_to_prob_space

    # Create a complex Gymnasium space
    gym_space = gym.spaces.Dict({
        'discrete': gym.spaces.Discrete(3),
        'multi': gym.spaces.MultiDiscrete([2, 3, 4]),
        'continuous': gym.spaces.Box(low=-1, high=1, shape=(2,))
    })

    # Convert to a probability space
    prob_space = convert_to_prob_space(gym_space)

Working with an environment:

.. code-block:: python

    import gymnasium as gym
    import torch as th
    from prob_spaces.converter import convert_to_prob_space

    # Create an environment
    env = gym.make('CartPole-v1')

    # Convert action space to a probability space
    action_prob_space = convert_to_prob_space(env.action_space)

    # Create a uniform distribution over actions
    probs = th.ones(env.action_space.n)
    dist = action_prob_space(probs)

    # Sample an action
    action = dist.sample().item()
    env.reset()
    # Take a step in the environment
    obs, reward, done, truncated, info = env.step(action)
