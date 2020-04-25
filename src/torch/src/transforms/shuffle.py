import torch


class Shuffle(object):
    r""" Shuffles point order. """

    def __init__(self):
        pass

    def __call__(self, data):
        pos_perm = torch.randperm(len(data.pos))
        data.pos = data.pos[pos_perm]
        data.graph_id = data.graph_id[pos_perm]

        y_perm = torch.randperm(len(data.y))
        data.y = data.y[y_perm]

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.scales)

