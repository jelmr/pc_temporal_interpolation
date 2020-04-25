import torch


def clamp(pc, clip):
    len_ = pc.size()[0]
    min_ = clip[0].repeat((len_, 1))
    max_ = clip[1].repeat((len_, 1))

    return torch.max(torch.min(pc, max_), min_)


class Jitter(object):
    r""" Adds a slight offset sampled from jitter_range, clipping values to clip_range.

    Args:
        jitter_range (float): the maximum jitter range as a factor relative to the
                min- and max values of the points.

                For example, if values in a certain dimension range from -2 to 5, and a
                jitter_range of 0.1 is specified, jitter will be same randomly between
                [-(5 - -2) * 0.1, (5 - -2) * 0.1] = [-0.7, 0.7]
        clip (tensor 2 x <num_features>): Specifies for each dimension where values should
                be clipped.
        uniform (bool): If true, one global min- max pair is used for all dimensions instead
                of a min- and a max per feature.

    """

    def __init__(self, jitter_range=0.002, clip=None, uniform=True):
        self.jitter_range = jitter_range
        self.clip = clip
        self.uniform = uniform

    def __call__(self, data):
        pc1_idx = data.graph_id == 0
        pc2_idx = data.graph_id == 1

        data.pos[pc1_idx] = self.jitter(data.pos[pc1_idx])
        data.pos[pc2_idx] = self.jitter(data.pos[pc2_idx])
        data.y = self.jitter(data.y)

        return data

    def jitter(self, pc):
        num_features = pc.size()[1]

        if self.uniform:
            min_ = pc.min().repeat(num_features)
            max_ = pc.max().repeat(num_features)
        else:
            min_ = pc.min(dim=0).values
            max_ = pc.max(dim=0).values

        diff = max_ - min_

        offsets = torch.rand(pc.size())
        offsets = offsets * 2 - 1  # add negative side
        offsets *= diff
        offsets *= self.jitter_range

        result = pc + offsets

        if self.clip is not None:
            result = clamp(result, self.clip)

        return result

    def __repr__(self):
        return '{}(range={})'.format(self.__class__.__name__, self.jitter_range)
