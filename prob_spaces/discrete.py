import torch as th
from gymnasium import spaces

from prob_spaces.dists.categorical import CategoricalDist, MaskedCategorical


class DiscreteDist(spaces.Discrete):
    def __call__(self, prob: th.Tensor, mask: th.Tensor = None) -> MaskedCategorical:
        probs = prob.reshape(self.n)  # type: ignore
        start = self.start
        mask = mask if mask is not None else th.ones_like(probs, dtype=th.bool, device=probs.device)
        dist = CategoricalDist(probs, mask=mask, start=start)
        return dist

    @classmethod
    def from_space(cls, space: spaces.Discrete) -> "DiscreteDist":
        return cls(
            n=space.n,
            start=space.start,
            # dtype=space.dtype,
        )
