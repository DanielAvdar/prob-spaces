from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from torchrl.modules import MaskedCategorical


class CategoricalDist(MaskedCategorical):
    def __init__(
        self,
        logits: Optional[th.Tensor] = None,
        probs: Optional[th.Tensor] = None,
        *,
        mask: th.Tensor = None,
        indices: th.Tensor = None,
        neg_inf: float = float("-inf"),
        padding_value: Optional[int] = None,
        start: int,
    ) -> None:
        super().__init__(logits, probs, mask=mask, indices=indices, neg_inf=neg_inf, padding_value=padding_value)
        self.start = start

    def sample(
        self,
        sample_shape: Optional[Union[th.Size, Sequence[int]]] = None,
    ) -> th.Tensor:
        sample = super().sample(sample_shape)
        if not isinstance(self.start, np.ndarray) or sum(self.start.shape) == 1:
            return sample + self.start
        else:
            return sample.reshape(self.probs.shape) + th.tensor(self.start)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value=value - self.start)


class DiscreteDist(spaces.Discrete):
    # space: spaces.Discrete

    def __call__(self, prob: th.Tensor, mask: th.Tensor = None) -> MaskedCategorical:
        probs = prob.reshape(-1, self.n)
        start = self.start
        mask = mask if mask is not None else th.ones_like(probs, dtype=th.bool)
        dist = CategoricalDist(probs, mask=mask, start=start)

        return dist


class MultiDiscreteDist(spaces.MultiDiscrete):
    def __call__(self, prob: th.Tensor, mask: th.Tensor = None) -> MaskedCategorical:
        probs = prob.reshape(*self.nvec.shape, -1, max(self.nvec - self.start))
        start = self.start
        mask = mask if mask is not None else th.ones_like(probs, dtype=th.bool)
        dist = CategoricalDist(probs, mask=mask, start=start)
        return dist
