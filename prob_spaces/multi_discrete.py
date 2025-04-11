from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray

from prob_spaces.dists.categorical import CategoricalDist, MaskedCategorical


class MultiDiscreteDist(spaces.MultiDiscrete):
    def __init__(
        self,
        nvec: NDArray[np.integer[Any]] | list[int],
        dtype: str | type[np.integer[Any]] = np.int64,
        seed: int | np.random.Generator | None = None,
        start: NDArray[np.integer[Any]] | list[int] | None = None,
    ):
        super().__init__(nvec, dtype, seed, start)
        self.internal_mask = self._internal_mask()

    @property
    def prob_last_dim(self) -> int:
        return int(np.max(self.nvec)) + 1

    def _internal_mask(self) -> NDArray[np.bool_]:
        prob_last_dim = self.prob_last_dim
        shape = (*self.nvec.shape, self.prob_last_dim)
        mask = np.zeros(shape=shape, dtype=np.bool)
        max_arrange = np.arange(start=0, stop=prob_last_dim)
        all_actions = np.zeros_like(mask, dtype=self.nvec.dtype)
        all_actions[..., :] = max_arrange
        diffs = np.abs(self.nvec)
        c_diffs = np.broadcast_to(diffs[..., np.newaxis], shape)
        mask[c_diffs > all_actions] = True

        return mask

    def __call__(self, prob: th.Tensor, mask: th.Tensor = None) -> MaskedCategorical:
        """
        Applies a transformation to the input probability tensor and optional mask, creating
        a `MaskedCategorical` distribution. The method reshapes the input probabilities
        to match the specified `nvec` dimensions, applies an optional mask for masking
        specific probabilities, and combines these with an internal mask. The result
        is used to create a `MaskedCategorical` distribution.

        :param prob: A tensor containing probabilities to be reshaped and used in
            constructing the distribution.
        :type prob: th.Tensor
        :param mask: An optional boolean tensor for masking specific probabilities
            before creating the distribution. Defaults to None.
        :type mask: th.Tensor, optional
        :return: A `MaskedCategorical` distribution object created with reshaped
            probabilities and combined masking information.
        :rtype: MaskedCategorical
        """
        probs = prob.reshape(*self.nvec.shape, self.prob_last_dim)
        start = self.start
        mask = mask if mask is not None else th.ones_like(probs, dtype=th.bool, device=probs.device)
        mask = th.logical_and(mask, th.tensor(self.internal_mask, dtype=th.bool, device=probs.device))
        dist = CategoricalDist(probs, mask=mask, start=start)
        return dist

    @classmethod
    def from_space(cls, space: spaces.MultiDiscrete) -> "MultiDiscreteDist":
        """Convert a gymnasium space to a MultiDiscreteDist."""
        return cls(nvec=space.nvec, dtype=space.dtype, start=space.start)  # type: ignore
