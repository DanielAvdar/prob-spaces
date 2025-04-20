"""Module for probability distributions over Discrete spaces."""

import torch as th
from gymnasium import spaces

from prob_spaces.dists.categorical import CategoricalDist, MaskedCategorical


class DiscreteDist(spaces.Discrete):
    """Probability distribution for Discrete spaces."""

    def __call__(self, prob: th.Tensor, mask: th.Tensor = None) -> MaskedCategorical:
        """Compute and return a masked categorical distribution.

        Compute a masked categorical distribution based on the given probability tensor and an
        optional mask. The distribution incorporates specific probabilities and constraints defined
        by the provided input.

        :param prob: A tensor representing the probabilities for each category.
        :param mask: A tensor specifying a mask to limit the valid categories. Defaults to a tensor
            of ones if not provided.
        :return: A MaskedCategorical distribution constructed with given probabilities, mask, and
            starting values.
        """
        probs = prob.reshape(self.n)  # type: ignore
        start = self.start
        mask = mask if mask is not None else th.ones_like(probs, dtype=th.bool, device=probs.device)
        dist = CategoricalDist(probs, mask=mask, start=start)
        return dist

    @classmethod
    def from_space(cls, space: spaces.Discrete) -> "DiscreteDist":
        """Create a DiscreteDist from a gymnasium Discrete space."""
        return cls(
            n=space.n,
            start=space.start,
            # dtype=space.dtype,
        )
