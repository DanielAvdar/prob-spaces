from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch as th
from numpy._typing import NDArray
from torchrl.modules import MaskedCategorical  # type: ignore


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
        start: int | np.integer[Any] | NDArray[np.integer[Any]] | list[int] = 0,
    ) -> None:
        super().__init__(logits, probs, mask=mask, indices=indices, neg_inf=neg_inf, padding_value=padding_value)
        self.start = start

    def sample(
        self,
        sample_shape: Optional[Union[th.Size, Sequence[int]]] = None,
    ) -> th.Tensor:
        sample = super().sample(sample_shape)
        exact_sample = self._calc_exact(sample)
        return exact_sample

    def rsample(
        self,
        sample_shape: Optional[Union[th.Size, Sequence[int]]] = None,
    ) -> th.Tensor:
        sample = super().rsample(sample_shape)
        exact_sample = self._calc_exact(sample)
        return exact_sample

    def _calc_exact(self, sample: th.Tensor) -> th.Tensor:
        if not isinstance(self.start, np.ndarray) or sum(self.start.shape) == 1:
            exact_sample = sample + self.start  # type: ignore
        else:
            exact_sample = sample.reshape(self.start.shape) + th.tensor(self.start)
        return exact_sample  # type: ignore

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value=value - self.start)  # type: ignore
