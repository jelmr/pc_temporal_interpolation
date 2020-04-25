import torch
import unittest
from transforms.center import Center
from torch_geometric.data import Data


class TestCenter(unittest.TestCase):

    def test_center(self):
        pc = torch.tensor([[-2., -2.], [-2., -1.], [-3., -2.], [-3., -1.]])
        pos = pc.repeat((2, 1))
        data = Data(pos=pos, y=pc.clone())

        res = Center()(data)

        self.assertAlmostEqual(res.pos.tolist(), [[0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5],
                                                  [0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5]])
        self.assertAlmostEqual(res.y.tolist(), [[0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5]])
