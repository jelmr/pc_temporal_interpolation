import torch
import unittest
from transforms.random_flip import RandomFlip
import numpy as np
from torch_geometric.data import Data


class TestRandomFlip(unittest.TestCase):

    def test_random_flip(self):
        pc_l = [[-1, -1, 1], [-1, 1, 1], [1, -1, -1], [1, 1, -1]]
        pc = torch.Tensor(pc_l)
        data = Data(pos=pc.clone(), y=pc.clone())

        res = RandomFlip(p=0.0, flip_x=False, flip_y=False, flip_z=False)(data)
        np.testing.assert_almost_equal(res.pos.tolist(), pc_l, decimal=4)
        np.testing.assert_almost_equal(res.y.tolist(), pc_l, decimal=4)

        pc_flipped_l = [[1, -1, 1], [1, 1, 1], [-1, -1, -1], [-1, 1, -1]]
        res = RandomFlip(p=1.0, flip_x=True, flip_y=False, flip_z=False)(data)
        np.testing.assert_almost_equal(res.pos.tolist(), pc_flipped_l, decimal=4)
        np.testing.assert_almost_equal(res.y.tolist(), pc_flipped_l, decimal=4)

        pc = torch.Tensor(pc_l)
        data = Data(pos=pc.clone(), y=pc.clone())
        pc_flipped_l = [[-1, 1, -1], [-1, -1, -1], [1, 1, 1], [1, -1, 1]]
        res = RandomFlip(p=1.0, flip_x=False, flip_y=True, flip_z=True)(data)
        np.testing.assert_almost_equal(res.pos.tolist(), pc_flipped_l, decimal=4)
        np.testing.assert_almost_equal(res.y.tolist(), pc_flipped_l, decimal=4)

