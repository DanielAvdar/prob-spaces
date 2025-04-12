# prob-spaces: Probability Distributions from Gymnasium Spaces

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/prob-spaces)](https://pypi.org/project/prob-spaces/)
[![version](https://img.shields.io/pypi/v/prob-spaces)](https://img.shields.io/pypi/v/prob-spaces)
[![License](https://img.shields.io/:license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![OS](https://img.shields.io/badge/ubuntu-blue?logo=ubuntu)
![OS](https://img.shields.io/badge/win-blue?logo=windows)
![OS](https://img.shields.io/badge/mac-blue?logo=apple)
[![Tests](https://github.com/DanielAvdar/prob-spaces/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielAvdar/prob-spaces/actions/workflows/ci.yml)
[![Code Checks](https://github.com/DanielAvdar/prob-spaces/actions/workflows/code-checks.yml/badge.svg)](https://github.com/DanielAvdar/prob-spaces/actions/workflows/code-checks.yml)
[![codecov](https://codecov.io/gh/DanielAvdar/prob-spaces/graph/badge.svg?token=N0V9KANTG2)](https://codecov.io/gh/DanielAvdar/prob-spaces)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Last Commit](https://img.shields.io/github/last-commit/DanielAvdar/prob-spaces/main)
**prob-spaces** is a Python package that allows you to create probability distributions from Gymnasium spaces.
It provides a simple and intuitive interface for working with various probability spaces in reinforcement learning
environments.

Key Features:

* Create probability distributions directly from Gymnasium spaces
* Support for common space types: Discrete, MultiDiscrete, Box, and Dict
* Seamless integration with PyTorch for sampling and computing log probabilities
* Support for masking operations to constrain valid actions

## Installation

### From PyPI

To install prob-spaces from PyPI:

```bash
pip install prob-spaces
```



### GPU Support

prob-spaces uses PyTorch, which can be installed with CUDA support for GPU acceleration.
The package configuration includes a PyTorch CUDA 12.4 index. To use a different CUDA version,
you may need to modify the PyTorch installation separately.

## Example Usage

Here's a simple example of how to use prob-spaces:

```python
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
```

## Documentation

[Documentation](https://prob-spaces.readthedocs.io/en/latest/) is available online and provides detailed information on how to use the package, including examples and API references.
