import torch


class UniformSample(object):
    r"""Uniformly samples a point cloud down to n points

    Args:
        n (int): The number of points to sample down to.
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, data):
        pc1 = data.pos[data.graph_id == 0, ...]
        pc2 = data.pos[data.graph_id == 1, ...]

        pc1_sampled = self.sample(pc1)
        pc2_sampled = self.sample(pc2)
        if "y" in data:
            data.y = self.sample(data.y)

        data.pos = torch.cat([pc1_sampled, pc2_sampled])

        graph_id = torch.zeros(2 * self.n)
        graph_id[self.n:] = 1
        data.graph_id = graph_id

        return data

    def sample(self, pc):
        n_rows = pc.shape[0]
        if n_rows <= self.n:
            return pc
        else:
            idx = torch.randperm(n_rows)[:self.n]
            return pc[idx, :]

    def __repr__(self):
        return '{}(n={})'.format(self.__class__.__name__, self.n)
