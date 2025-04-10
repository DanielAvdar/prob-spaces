from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch as th
import torch.distributions as D  # noqa: N812
from numpy._typing import NDArray
from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits


class MaskedCategorical(D.Categorical):
    """MaskedCategorical distribution.

    Reference:
    https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/masked/MaskedCategorical

    Args:
        logits (torch.Tensor): event log probabilities (unnormalized)
        probs (torch.Tensor): event probabilities. If provided, the probabilities
            corresponding to masked items will be zeroed and the probability
            re-normalized along its last dimension.

    Keyword Args:
        mask (torch.Tensor): A boolean mask of the same shape as ``logits``/``probs``
            where ``False`` entries are the ones to be masked. Alternatively,
            if ``sparse_mask`` is True, it represents the list of valid indices
            in the distribution. Exclusive with ``indices``.
        indices (torch.Tensor): A dense index tensor representing which actions
            must be taken into account. Exclusive with ``mask``.
        neg_inf (:obj:`float`, optional): The log-probability value allocated to
            invalid (out-of-mask) indices. Defaults to -inf.
        padding_value: The padding value in the mask tensor. When
            sparse_mask == True, the padding_value will be ignored.

        >>> torch.manual_seed(0)
        >>> logits = torch.randn(4) / 100  # almost equal probabilities
        >>> mask = torch.tensor([True, False, True, True])
        >>> dist = MaskedCategorical(logits=logits, mask=mask)
        >>> sample = dist.sample((10,))
        >>> print(sample)  # no `1` in the sample
        tensor([2, 3, 0, 2, 2, 0, 2, 0, 2, 2])
        >>> print(dist.log_prob(sample))
        tensor([-1.1203, -1.0928, -1.0831, -1.1203, -1.1203, -1.0831, -1.1203, -1.0831,
                -1.1203, -1.1203])
        >>> print(dist.log_prob(torch.ones_like(sample)))
        tensor([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf])
        >>> # with probabilities
        >>> prob = torch.ones(10)
        >>> prob = prob / prob.sum()
        >>> mask = torch.tensor([False] + 9 * [True])  # first outcome is masked
        >>> dist = MaskedCategorical(probs=prob, mask=mask)
        >>> print(dist.log_prob(torch.arange(10)))
        tensor([   -inf, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972,
                -2.1972, -2.1972])
    """

    @lazy_property
    def logits(self):  # type: ignore
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self):  # type: ignore
        return logits_to_probs(self.logits)

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        *,
        mask: torch.Tensor = None,
        indices: torch.Tensor = None,
        neg_inf: float = float("-inf"),
        padding_value: Optional[int] = None,
    ) -> None:
        if not ((mask is None) ^ (indices is None)):
            raise ValueError(f"A ``mask`` or some ``indices`` must be provided for {type(self)}, but not both.")
        if mask is None:
            mask = indices
            sparse_mask = True
        else:
            sparse_mask = False

        if probs is not None:
            if logits is not None:
                raise ValueError("Either `probs` or `logits` must be specified, but not both.")
            # unnormalized logits
            probs = probs.clone()
            probs[~mask] = 0
            probs = probs / probs.sum(-1, keepdim=True)
            logits = probs.log()
        num_samples = logits.shape[-1]
        logits = self._mask_logits(
            logits,
            mask,
            neg_inf=neg_inf,
            sparse_mask=sparse_mask,
            padding_value=padding_value,
        )
        self.neg_inf = neg_inf
        self._mask = mask
        self._sparse_mask = sparse_mask
        self._padding_value = padding_value
        super().__init__(logits=logits)
        self.num_samples = num_samples

    def sample(self, sample_shape: Optional[Union[torch.Size, Sequence[int]]] = None) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        else:
            sample_shape = torch.Size(sample_shape)

        ret = super().sample(sample_shape)
        if not self._sparse_mask:
            return ret

        size = ret.size()
        outer_dim = sample_shape.numel()
        inner_dim = self._mask.shape[:-1].numel()
        idx_3d = self._mask.expand(outer_dim, inner_dim, -1)
        ret = idx_3d.gather(dim=-1, index=ret.view(outer_dim, inner_dim, 1))
        return ret.reshape(size)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if not self._sparse_mask:
            return super().log_prob(value)

        idx_3d = self._mask.view(1, -1, self._num_events)
        val_3d = value.view(-1, idx_3d.size(1), 1)
        mask = idx_3d == val_3d
        idx = mask.int().argmax(dim=-1, keepdim=True)
        ret = super().log_prob(idx.view_as(value))
        # Fill masked values with neg_inf.
        ret = ret.view_as(val_3d)
        ret = ret.masked_fill(torch.logical_not(mask.any(dim=-1, keepdim=True)), self.neg_inf)
        return ret.resize_as(value)

    @staticmethod
    def _mask_logits(
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        neg_inf: float = float("-inf"),
        sparse_mask: bool = False,
        padding_value: Optional[int] = None,
    ) -> torch.Tensor:
        if mask is None:
            return logits

        if not sparse_mask:
            return logits.masked_fill(~mask, neg_inf)

        if padding_value is not None:
            padding_mask = mask == padding_value
            if padding_value != 0:
                # Avoid invalid indices in mask.
                mask = mask.masked_fill(padding_mask, 0)
        logits = logits.gather(dim=-1, index=mask)
        if padding_value is not None:
            logits.masked_fill_(padding_mask, neg_inf)
        return logits

    @property
    def deterministic_sample(self):  # type: ignore
        return self.mode


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

    @property
    def th_start(self) -> th.Tensor:
        return th.tensor(self.start, device=self.probs.device)

    def sample(
        self,
        sample_shape: Optional[Union[th.Size, Sequence[int]]] = None,
    ) -> th.Tensor:
        sample = super().sample(sample_shape)
        exact_sample = self._calc_exact(sample, sample_shape)
        return exact_sample

    def rsample(
        self,
        sample_shape: Optional[Union[th.Size, Sequence[int]]] = None,
    ) -> th.Tensor:
        sample = super().rsample(sample_shape)  # type: ignore
        exact_sample = self._calc_exact(sample, sample_shape)
        return exact_sample

    def _calc_exact(
        self,
        sample: th.Tensor,
        sample_shape: Optional[Union[th.Size, Sequence[int]]],
    ) -> th.Tensor:
        if not isinstance(self.start, np.ndarray) or sum(self.start.shape) == 1:
            exact_sample = sample + self.start  # type: ignore
        else:
            shape = self.start.shape if sample_shape is None else (*sample_shape, *self.start.shape)
            exact_sample = sample.reshape(shape) + self.th_start
        return exact_sample  # type: ignore

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value=value - self.th_start)
