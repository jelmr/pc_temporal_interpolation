import random


class RandomReverseFrames(object):
    r""" With probability p switches around the first- and last frame of a triple.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        assert "graph_id" in data

        if random.random() < self.p:
            data.graph_id = 1 - data.graph_id

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
