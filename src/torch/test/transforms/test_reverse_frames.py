import torch
import unittest
from transforms.random_reverse_frames import RandomReverseFrames
import numpy as np
from torch_geometric.data import Data


class TestRandomReverseFrames(unittest.TestCase):

    def test_random_reverse_frames(self):
        pc1_l = [[1,1,1], [2,2,2]]
        pc1 = torch.tensor(pc1_l)

        pc2_l  = [[3,3,3], [4,4,4]]
        pc2 = torch.tensor(pc2_l)

        pc = torch.cat([pc1, pc2])
        graph_id = torch.tensor([0,0,1,1])
        data = Data(pos=pc.clone(), y=pc1.clone())
        data.graph_id = graph_id

        res = RandomReverseFrames(p=0.)(data)
        np.testing.assert_almost_equal(res.pos[res.graph_id == 0].tolist(), pc1, decimal=4)
        np.testing.assert_almost_equal(res.pos[res.graph_id == 1].tolist(), pc2, decimal=4)
        np.testing.assert_almost_equal(res.y.tolist(), pc1, decimal=4)

        pc = torch.cat([pc1, pc2])
        graph_id = torch.tensor([0,0,1,1])
        data = Data(pos=pc.clone(), y=pc1.clone())
        data.graph_id = graph_id
        res = RandomReverseFrames(p=1.)(data)
        np.testing.assert_almost_equal(res.pos[res.graph_id == 0].tolist(), pc2, decimal=4)
        np.testing.assert_almost_equal(res.pos[res.graph_id == 1].tolist(), pc1, decimal=4)
        np.testing.assert_almost_equa(res.y.tolist(), pc1, decimal=4)

