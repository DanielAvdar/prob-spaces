from gymnasium import spaces


class DictDist(spaces.Dict):
    def __call__(self, prob: dict) -> dict: ...

    @classmethod
    def from_space(cls, space: spaces.Dict) -> "DictDist":
        """Convert a gymnasium space to a MultiDiscreteDist."""
        raise NotImplementedError()
        # dict_dist = dict()
        # for key, value in space.items():
        #     if isinstance(value, spaces.MultiDiscrete):
        #         dict_dist[key] = MultiDiscreteDist.from_space(value)
        #     if isinstance(value, spaces.Discrete):
        #         dict_dist[key] = DiscreteDist.from_space(value)
        #     if isinstance(value, spaces.Box):
        #         dict_dist[key] = BoxDist.from_space(value)
        #     if isinstance(value, spaces.Dict):
        #         dict_dist[key] = DictDist.from_space(value)
        # return cls(dict_dist)
