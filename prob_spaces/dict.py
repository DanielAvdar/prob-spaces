"""Module for probability distributions over Dict spaces."""

from gymnasium import spaces


class DictDist(spaces.Dict):
    """Probability distribution for Dict spaces."""

    def __call__(self, prob: dict, mask: dict = None) -> dict:
        """Create a dict of distributions based on input probabilities.

        Args:
            prob: Dictionary of probability tensors for each space.
            mask: Optional dictionary of masks for each space.

        Returns:
            Dictionary of distribution objects.

        """
        dist_dict = {}
        mask = mask or {}

        for key, s in self.spaces.items():
            if isinstance(s, spaces.Discrete) or isinstance(s, spaces.MultiDiscrete):
                space_mask = mask.get(key, None) if isinstance(mask, dict) else None
                dist_dict[key] = s(prob[key], space_mask)  # type: ignore
            else:
                dist_dict[key] = s(  # type: ignore
                    prob[key][0],
                    prob[key][1],
                )

        return dist_dict
