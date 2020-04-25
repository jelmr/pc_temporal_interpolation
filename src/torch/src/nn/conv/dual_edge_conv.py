import torch
from torch_geometric.utils import scatter_

from torch_geometric.nn.inits import reset


class DualEdgeConv(torch.nn.Module):
    r"""
        TODO: Doc

    Args:
        nn (nn.Sequential): Neural network to be applied to generated edge features
        aggr (String): The aggregation operator to use ("add", "mean", "max", default: "max")
    """

    def __init__(self, nn, aggr='max'):
        super(DualEdgeConv, self).__init__()
        self.nn = nn
        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x1, edge_index1, x2, edge_index2):
        """ Perform DualEdgeConv operation.

        :param x1: point cloud 1
        :param edge_index1: edge set from x1 -> x1 describing nearest neighbours
        :param x2: point cloud 2
        :param edge_index2: edge set from x1 -> x2 describing nearest neighbours
        :return:
        """

        # Neighbourhood features pc
        row1, col1 = edge_index1
        x1 = x1.unsqueeze(-1) if x1.dim() == 1 else x1

        out1 = torch.cat([x1[row1], x1[col1] - x1[row1]], dim=1)
        out1 = self.nn(out1)
        out1 = scatter_(self.aggr, out1, row1, dim_size=x1.size(0))

        # Neighbourhood features pc2
        row2, col2 = edge_index2
        x2 = x2.unsqueeze(-1) if x2.dim() == 1 else x2
        out2 = torch.cat([x1[row2], x2[col2] - x1[row2]], dim=1)
        out2 = self.nn(out2)
        out2 = scatter_(self.aggr, out2, row2, dim_size=x2.size(0))

        out = torch.cat([out1, out2], dim=-1)

        return out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
