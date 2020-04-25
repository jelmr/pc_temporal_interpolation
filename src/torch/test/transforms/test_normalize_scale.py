import torch
import unittest
import numpy as np
from transforms.normalize_scale import NormalizeScale
from torch_geometric.data import Data


class TestNormalizeScale(unittest.TestCase):

    def test_normalize_scale(self):
        pc = torch.tensor([[-2., -2.], [-2., -1.], [-3., -2.], [-3., -1.]])
        pos = pc.repeat((2, 1))
        data = Data(pos=pos, y=pc.clone())

        res = NormalizeScale()(data)

        np.testing.assert_almost_equal(res.pos.tolist(), [[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0],
                                                  [1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]], decimal=5)
        np.testing.assert_almost_equal(res.y.tolist(), [[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]], decimal=5)
