import torch
import unittest
from transforms.random_scale import RandomScale
import numpy as np
from torch_geometric.data import Data


class TestRandomScale(unittest.TestCase):

    def test_random_scale(self):
        pc_l = [[-1, -1, 1], [-1, 1, 1], [1, -1, -1], [1, 1, -1]]
        pc = torch.Tensor(pc_l)
        data = Data(pos=pc.clone(), y=pc.clone())

        res = RandomScale(scales=[1., 1.])(data)
        np.testing.assert_almost_equal(res.pos.tolist(), pc_l, decimal=4)
        np.testing.assert_almost_equal(res.y.tolist(), pc_l, decimal=4)


        pc = torch.Tensor(pc_l)
        data = Data(pos=pc.clone(), y=pc.clone())
        res = RandomScale(scales=[2., 2.])(data)
        pc_scaled_l = [[-2, -2, 2], [-2, 2, 2], [2, -2, -2], [2, 2, -2]]
        np.testing.assert_almost_equal(res.pos.tolist(), pc_scaled_l, decimal=4)
        np.testing.assert_almost_equal(res.y.tolist(), pc_scaled_l, decimal=4)

