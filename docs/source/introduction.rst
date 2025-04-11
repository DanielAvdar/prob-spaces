.. _introduction:

Introduction
===========

Overview
--------

**prob-spaces** extends Gymnasium spaces to create probability distributions that can be used
in reinforcement learning and other machine learning applications. It provides a seamless
interface between Gymnasium spaces and probability distributions.

Key Features
-----------

* Create probability distributions directly from Gymnasium spaces
* Support for common space types: Discrete, MultiDiscrete, Box, and Dict
* Seamless integration with PyTorch for sampling and computing log probabilities
* Support for masking operations to constrain valid actions

Use Cases
---------

prob-spaces is particularly useful for:

* Reinforcement learning algorithms that require probability distributions over action spaces
* Implementing stochastic policies in RL agents
* Working with complex action spaces in Gymnasium environments
* Converting between different types of spaces and distributions

Example Usage
------------

Here's a simple example of how to use prob-spaces:

.. code-block:: python

    import gymnasium as gym
    import torch as th
    from prob_spaces.converter import convert_to_prob_space

    # Create a Gymnasium space
    action_space = gym.spaces.Discrete(5)

    # Convert to a probability space
    prob_space = convert_to_prob_space(action_space)

    # Create a probability distribution
    probs = th.ones(5)  # Uniform distribution
    dist = prob_space(probs)

    # Sample from the distribution
    action = dist.sample()

    # Compute log probability
    log_prob = dist.log_prob(action)
