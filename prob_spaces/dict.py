from gymnasium import spaces


class DictDist(spaces.Dict):
    def __call__(self, prob: dict, mask: dict = None) -> dict:
        """
        Create a dict of distributions based on input probabilities.

        Args:
            prob: Dictionary of probability tensors for each space
            mask: Optional dictionary of masks for each space

        Returns:
            Dictionary of distribution objects
        """
        dist_dict = {}
        mask = mask or {}

        for key, s in self.spaces.items():
            space_mask = mask.get(key, None)
            dist_dict[key] = s(prob[key], mask=space_mask)  # type: ignore #todo change into prob_spaces type

        return dist_dict
