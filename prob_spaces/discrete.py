from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray
from torchrl.modules import MaskedCategorical  # type: ignore

from prob_spaces.dists.categorical import CategoricalDist


class DiscreteDist(spaces.Discrete):
    def __call__(self, prob: th.Tensor, mask: th.Tensor = None) -> MaskedCategorical:
        probs = prob.reshape(-1, self.n)  # type: ignore
        start = self.start
        mask = mask if mask is not None else th.ones_like(probs, dtype=th.bool)
        dist = CategoricalDist(probs, mask=mask, start=start)

        return dist


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
        max_arrange = np.arange(start=prob_last_dim - 1, stop=-1, step=-1)
        max_arrange = np.arange(start=0, stop=prob_last_dim)
        all_actions = np.zeros_like(mask, dtype=self.nvec.dtype)
        all_actions[..., :] = max_arrange
        diffs = np.abs(self.nvec)
        c_diffs = np.broadcast_to(diffs[..., np.newaxis], shape)
        mask[c_diffs > all_actions] = True

        return mask

    def __call__(self, prob: th.Tensor, mask: th.Tensor = None) -> MaskedCategorical:
        probs = prob.reshape(*self.nvec.shape, self.prob_last_dim)
        start = self.start
        mask = mask if mask is not None else th.ones_like(probs, dtype=th.bool)
        mask = th.logical_and(mask, th.tensor(self.internal_mask, dtype=th.bool))
        dist = CategoricalDist(probs, mask=mask, start=start)
        return dist
