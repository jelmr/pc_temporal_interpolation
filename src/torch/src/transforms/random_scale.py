import random


class RandomScale(object):
    r"""Scales node positions by a factor sampled from the specified interval

    Args:
        scale (tuple): range from which scaling factor is sampled
    """

    def __init__(self, scales=(0.9, 1.1)):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data.pos = data.pos * scale
        data.y = data.y * scale
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.scales)

