"""Module for probability distributions over Tuple spaces."""

from gymnasium import spaces


class TupleDist(spaces.Tuple):
    """Probability distribution for Tuple spaces."""

    def __call__(self, prob: tuple, mask: tuple = None) -> tuple:
        """Create a tuple of distributions based on input probabilities.

        Args:
            prob: Tuple of probability tensors for each space.
            mask: Optional tuple of masks for each space.

        Returns:
            Tuple of distribution objects.

        """
        dist_list = []
        mask = mask or (None,) * len(self.spaces)

        for i, s in enumerate(self.spaces):
            space_mask = mask[i]

            if isinstance(s, (spaces.Discrete, spaces.MultiDiscrete)):
                dist_list.append(s(prob[i], space_mask))  # type: ignore
            else:
                dist_list.append(s(prob[i][0], prob[i][1]))  # type: ignore

        return tuple(dist_list)
