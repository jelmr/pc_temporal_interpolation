import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BatchNorm1d
import torch.nn
from torch_geometric.utils import scatter_

from torch_geometric.nn.inits import reset


class Linear(torch.nn.Module):
    r"""
        TODO: Doc

    Args:
        nn (nn.Sequential): Neural network to be applied to generated edge features
        aggr (String): The aggregation operator to use ("add", "mean", "max", default: "max")
    """

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 activation_fn=ReLU(),
                 bn=False):
        super(Linear, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lin = torch.nn.Linear(num_inputs, num_outputs)
        self.activation_fn=activation_fn

        self.reset_parameters()

        self.bn = bn
        if bn:
            # self.bn = BatchNorm1d(num_features=num_outputs)
            self.bn = False

    def reset_parameters(self):
        reset(self.lin)

    def forward(self, data):
        data = self.lin(data)

        if self.activation_fn:
            data = self.activation_fn(data)

        if self.bn:
            data = self.bn(data)

        return data

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.num_inputs, self.num_outputs)
