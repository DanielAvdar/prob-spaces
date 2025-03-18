from typing import Any, Sequence, SupportsFloat

import numpy as np
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import AffineTransform, SigmoidTransform


class BoxDist(spaces.Box):
    def __init__(
        self,
        low: SupportsFloat | NDArray[Any],
        high: SupportsFloat | NDArray[Any],
        shape: Sequence[int] | None = None,
        dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
        dist: None | th.distributions.Distribution = None,
    ):
        super().__init__(low, high, shape, dtype, seed)
        t_low = th.tensor(low)
        t_high = th.tensor(high)
        range_value = t_high - t_low
        offset = t_low
        self.base_dist = dist or th.distributions.Normal
        transforms = []
        if self.base_dist != th.distributions.Beta:
            transforms.append(SigmoidTransform())
        transforms.append(AffineTransform(loc=offset, scale=range_value, event_dim=1))
        self.transforms = transforms

    def __call__(self, loc, scale) -> th.distributions.Distribution:
        dist = self.base_dist(loc + 0.001, scale + 0.001, validate_args=True)
        transformed_dist = TransformedDistribution(dist, self.transforms, validate_args=True)
        return transformed_dist
