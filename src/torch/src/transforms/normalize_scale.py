from transforms.center import Center


class NormalizeScale(object):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`.
    """

    def __init__(self):
        self.center = Center()

    def __call__(self, data):
        data = self.center(data)

        if "y" in data:
            max_ = max(data.pos[..., :3].abs().max(), data.y[..., :3].abs().max())
        else:
            max_ = data.pos[..., :3].abs().max()
        scale = (1 / max_) * 0.999999
        data.pos[..., :3] = data.pos[..., :3] * scale
        if "y" in data:
            data.y[..., :3] = data.y[..., :3] * scale

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
