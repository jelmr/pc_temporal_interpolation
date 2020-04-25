import random


class RandomFlip(object):
    r""" With independent probability p flips the point clouds
         along the specified axis.

    Args:
        n (int): The number of points to sample down to.
    """

    def __init__(self, flip_x=True, flip_y=False, flip_z=True, p=0.5):
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.flip_z = flip_z
        self.p = p

    def __call__(self, data):
        if self.flip_x:
            self.flip(data, 0)
        if self.flip_y:
            self.flip(data, 1)
        if self.flip_z:
            self.flip(data, 2)

        return data

    def flip(self, data, dim):
        if random.random() < self.p:
            data.pos[..., dim] = -data.pos[..., dim]
            data.y[..., dim] = -data.y[..., dim]



    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)
