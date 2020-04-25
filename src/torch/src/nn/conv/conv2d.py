import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BatchNorm1d
import torch.nn
from torch_geometric.utils import scatter_

from torch_geometric.nn.inits import reset


class Conv2d(torch.nn.Module):
    r"""
        TODO: Doc

    Args:
        nn (nn.Sequential): Neural network to be applied to generated edge features
        aggr (String): The aggregation operator to use ("add", "mean", "max", default: "max")
    """

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 activation_fn=ReLU(),
                 bn=False):
        super(Conv2d, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.conv = torch.nn.Conv1d(num_inputs, num_outputs, kernel_size, stride, padding)
        self.activation_fn = activation_fn

        self.reset_parameters()

        if bn:
            self.bn = BatchNorm1d(num_features=num_outputs)
        else:
            self.bn = False

    def reset_parameters(self):
        reset(self.conv)

    def forward(self, data):
        data = self.conv(data.unsqueeze(-1)).squeeze()

        if self.bn:
            data = self.bn(data)

        if self.activation_fn:
            data = self.activation_fn(data)

        return data

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.num_inputs, self.num_outputs)
