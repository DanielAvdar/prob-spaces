"""Module for probability distributions over Box spaces."""

from typing import Any, Sequence, SupportsFloat, Type

import numpy as np
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import AffineTransform, SigmoidTransform


class BoxDist(spaces.Box):
    """Probability distribution for Box spaces."""

    def __init__(
        self,
        low: SupportsFloat | NDArray[Any],
        high: SupportsFloat | NDArray[Any],
        shape: Sequence[int] | None = None,
        dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
        dist: None | Type[th.distributions.Distribution] = None,
    ):
        """Initialize BoxDist with bounds, shape, dtype, and base distribution."""
        super().__init__(low, high, shape, dtype, seed)
        self.base_dist = dist or th.distributions.Normal

    def transforms(self, device: th.device) -> list:
        """Return list of transforms to map base distribution to Box bounds.

        Returns
        -------
        list
            List of transforms to map the base distribution to Box bounds.

        """
        t_low = th.tensor(self.low, device=device)
        t_high = th.tensor(self.high, device=device)
        range_value = t_high - t_low
        offset = t_low
        transforms: list = []
        if self.base_dist != th.distributions.Beta:
            transforms.append(SigmoidTransform())
        transforms.append(AffineTransform(loc=offset, scale=range_value, event_dim=1))
        return transforms

    def __call__(self, loc: th.Tensor, scale: th.Tensor) -> th.distributions.Distribution:
        """Generate a transformed probability distribution.

        Construct a base distribution, apply a sequence of transformations, and return the resulting
        transformed distribution.

        :param loc: A tensor specifying the location parameters for the base distribution.
        :param scale: A tensor specifying the scale parameters for the base distribution.
        :return: A transformed distribution object derived from the specified base distribution and
            transformations.

        Returns
        -------
        th.distributions.Distribution
            A transformed distribution object derived from the specified base distribution and transformations.

        """
        dist = self.base_dist(loc, scale, validate_args=True)  # type: ignore
        transforms = self.transforms(loc.device)
        transformed_dist = TransformedDistribution(dist, transforms, validate_args=True)
        return transformed_dist

    @classmethod
    def from_space(cls, space: spaces.Box) -> "BoxDist":
        """Create a BoxDist from a gymnasium Box space.

        Returns
        -------
        BoxDist
            An instance of BoxDist created from the given gymnasium Box space.

        """
        low = space.low
        high = space.high
        dtype = space.dtype
        shape = space.shape
        return cls(low=low, high=high, shape=shape, dtype=dtype)  # type: ignore
