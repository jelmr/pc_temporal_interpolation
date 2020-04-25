import torch

class Center(object):
    r"""Centers node positions around the origin."""

    def __call__(self, data):
        if "y" in data:
            all = torch.cat([data.pos, data.y], dim=0)
        else:
            all = data.pos

        mean = all.mean(dim=0, keepdim=True)
        data.pos[..., :3] -= mean[..., :3]
        if "y" in data:
            data.y[..., :3] -= mean[..., :3]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
