import torch
import unittest
from transforms.uniform_sample import UniformSample
from torch_geometric.data import Data


class TestUniformSample(unittest.TestCase):

    def test_uniform_sample(self):
        pc1 = torch.rand(1000, 6)
        pc2 = torch.rand(1500, 6)
        target = torch.rand(600, 6)
        graph_id = torch.zeros(2500)
        graph_id[1000:] = 1
        pc = torch.cat([pc1, pc2])

        data = Data(pos=pc, y=target)
        data.graph_id = graph_id
        res = UniformSample(n=200)(data)

        self.assertEqual(res.pos[res.graph_id == 0].shape, (200, 6))
        self.assertEqual(res.pos[res.graph_id == 1].shape, (200, 6))
        self.assertEqual(res.y.shape, (200, 6))

