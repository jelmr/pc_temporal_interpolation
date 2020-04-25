import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from nn.conv.dual_edge_conv import DualEdgeConv
import unittest


class TestDualEdgeConv(unittest.TestCase):

    def test_dual_edge_conv_conv(self):

        in_channels, out_channels = (16, 32)
        edge_index1 = torch.tensor([[0, 0, 0, 1, 2, 3],
                                    [1, 2, 3, 0, 0, 0]])

        edge_index2 = torch.tensor([[1, 2, 3, 3, 1, 2],
                                    [3, 1, 2, 1, 2, 3]])

        num_nodes_1 = edge_index1.max().item() + 1
        num_nodes_2 = edge_index2.max().item() + 1

        x1 = torch.randn((num_nodes_1, in_channels))
        x2 = torch.randn((num_nodes_2, in_channels))

        nn = Seq(Lin(2 * in_channels, 32), ReLU(), Lin(32, out_channels))
        conv = DualEdgeConv(nn)
        assert conv.__repr__() == (
            'DualEdgeConv(nn=Sequential(\n'
            '  (0): Linear(in_features=32, out_features=32, bias=True)\n'
            '  (1): ReLU()\n'
            '  (2): Linear(in_features=32, out_features=32, bias=True)\n'
            '))')
        assert conv(x1, edge_index1, x2, edge_index2).size() == (num_nodes_1, out_channels * 2)

