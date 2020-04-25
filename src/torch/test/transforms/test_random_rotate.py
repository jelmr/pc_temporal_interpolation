import torch
import unittest
from transforms.random_rotate import RandomRotate
import numpy as np
from torch_geometric.data import Data


class TestRandomRotate(unittest.TestCase):

    def test_random_rotate_(self):
        pc_l = [[-1, -1, 1], [-1, 1, 1], [1, -1, -1], [1, 1, -1]]
        pc = torch.Tensor(pc_l)
        data = Data(pos=pc.clone(), y=pc.clone())

        # num_features=3 testing
        res = RandomRotate(degree_range=(0.,0.), axis=0)(data)
        np.testing.assert_almost_equal(res.pos.tolist(), pc_l, decimal=4)
        np.testing.assert_almost_equal(res.y.tolist(), pc_l, decimal=4)

        pc_rotated_l = [[-1, 1, -1], [-1, -1, -1], [1, 1, 1], [1, -1, 1]]
        res = RandomRotate(degree_range=(180.,180.), axis=0)(data)
        np.testing.assert_almost_equal(res.pos.tolist(), pc_rotated_l, decimal=4)
        np.testing.assert_almost_equal(res.y.tolist(), pc_rotated_l, decimal=4)


        # num_features>3 testing
        pc_l = [[-1, -1, 1, 4], [-1, 1, 1, 5], [1, -1, -1, 6], [1, 1, -1, 7]]
        pc = torch.Tensor(pc_l)
        data = Data(pos=pc.clone(), y=pc.clone())

        res = RandomRotate(degree_range=(0.,0.), axis=0)(data)
        np.testing.assert_almost_equal(res.pos.tolist(), pc_l, decimal=4)
        np.testing.assert_almost_equal(res.y.tolist(), pc_l, decimal=4)

        pc_rotated_l = [[-1, 1, -1, 4], [-1, -1, -1, 5], [1, 1, 1, 6], [1, -1, 1, 7]]
        res = RandomRotate(degree_range=(180.,180.), axis=0)(data)
        np.testing.assert_almost_equal(res.pos.tolist(), pc_rotated_l, decimal=4)
        np.testing.assert_almost_equal(res.y.tolist(), pc_rotated_l, decimal=4)
