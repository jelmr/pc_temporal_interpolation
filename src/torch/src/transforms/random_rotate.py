import math
import random

import torch


class RandomRotate(object):
    r""" Rotates along the specified axis.

    Args:
        degree_range (tuple): Range from which rotation degree is sampled
        axis (int, optional): The rotation axis
    """

    def __init__(self, degree_range=(0, 360), axis=0):
        self.degree_range = degree_range
        self.axis = axis

    def __call__(self, data):

        degree = math.pi * random.uniform(*self.degree_range) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        matrix = torch.tensor(matrix)

        data.pos[..., :3] = torch.mm(data.pos[..., :3], matrix.to(data.pos.dtype).to(data.pos.device))
        data.y[..., :3] = torch.mm(data.y[..., :3], matrix.to(data.y.dtype).to(data.y.device))

        return data

    def __repr__(self):
        return '{}({}, axis={})'.format(self.__class__.__name__, self.degree_range,
                                        self.axis)

