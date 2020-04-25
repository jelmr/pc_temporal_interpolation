import torch
import unittest
from transforms.jitter import Jitter
from torch_geometric.data import Data
import numpy as np


class TestJitter(unittest.TestCase):

    def test_jitter(self):
        pc1 = torch.rand(1000, 6)
        pc2 = torch.rand(1500, 6)
        target = torch.rand(600, 6)
        graph_id = torch.zeros(2500)
        graph_id[1000:] = 1
        pc = torch.cat([pc1, pc2])

        data = Data(pos=pc.clone(), y=target.clone())
        data.graph_id = graph_id

        range_max = 0.0002
        res = Jitter(jitter_range=range_max)(data)

        # Jitters not too large
        diff_pos = res.pos - pc
        np.testing.assert_array_less(np.abs(diff_pos), range_max)

        # Shape same
        self.assertEqual(res.pos.shape, pc.shape)

        # Test clip
        clip_range = torch.tensor([
            [.4, .3, .2, .1, .1, .1],
            [.5, .6, .7, .8, .9, .95]
        ])

        data = Data(pos=pc.clone(), y=target.clone())
        data.graph_id = graph_id
        res = Jitter(jitter_range=range_max, clip=clip_range)(data)

        for i in range(6):
            self.assertTrue((data.pos[..., i] <= clip_range[1, i]).all())
            self.assertTrue((clip_range[0, i] <= data.pos[..., i] ).all())


