import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel
from nn.conv import Linear, Conv2d
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn.init as init
from torch_geometric.nn import PointConv, fps, radius, EdgeConv, knn_graph, global_max_pool
from torch_cluster import knn
from nn.conv.dual_edge_conv import DualEdgeConv


class PCI_Net(BaseModel):

    def __init__(self, k):
        super(PCI_Net, self).__init__()

        self.k = k

        conv1_out = 64
        nn = Seq(
            Linear(12, conv1_out, bn=True),
            Linear(conv1_out, conv1_out, bn=True),
            Linear(conv1_out, conv1_out, bn=True),
        )

        #nn = Seq(Conv2d(12, conv1_out), ReLU(), Conv2d(conv1_out, conv1_out), ReLU(), Conv2d(conv1_out, conv1_out), ReLU())
        self.conv1 = DualEdgeConv(nn, aggr='max')

        conv2_out = 256
        nn = Seq(
            Linear(2*2*conv1_out, 128, bn=True),
            Linear(128, 128, bn=True),
            Linear(128, conv2_out, bn=True)
        )
        # nn = Seq(Conv2d(2*2*conv1_out, 128), ReLU(), Conv2d(128, 128), ReLU(), Conv2d(128, conv2_out), ReLU())
        self.conv2 = EdgeConv(nn, aggr='max')

        conv3_out = 1024
        #self.conv3 = Conv2d(2*2*conv1_out+conv2_out, conv3_out, bn=True)
        self.conv3 = Linear(2*2*conv1_out+conv2_out, conv3_out, bn=True)

        # Convs at end
        self.conv4 = Conv2d(conv3_out+2*2*conv1_out+conv2_out, 256, bn=True)
        self.conv5 = Conv2d(256, 256, bn=True)
        self.conv6 = Conv2d(256, 128, bn=False, activation_fn=None)
        self.conv7 = Conv2d(128, 6, bn=False, activation_fn=None)
        # self.conv4 = Linear(conv3_out+2*2*conv1_out+conv2_out, 256, bn=True)
        # self.conv5 = Linear(256, 256, bn=True)
        # self.conv6 = Linear(256, 128, bn=False, activation_fn=None)
        # self.conv7 = Linear(128, 6, bn=False, activation_fn=None)

        self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, pc1, pc2, pc1_batch, pc2_batch):
        num_points = pc1.shape[1]

        # Center both point clouds around origin
        pcs = torch.cat([pc1, pc2])[..., :3]
        centroids = torch.mean(pcs, dim=0)
        pc1[..., :3] -= centroids
        pc2[..., :3] -= centroids

        pc1_centroids = torch.mean(pc1, dim=0)
        pc2_centroids = torch.mean(pc1, dim=0)
        pc1 -= pc1_centroids
        pc2 -= pc2_centroids
        edge_index1 = knn_graph(pc1, k=self.k, batch=pc1_batch)
        edge_index2 = knn(pc2, pc1, self.k, batch_x=pc2_batch, batch_y=pc1_batch) # TODO: Sort out batches
        # print("net1")
        pc1 += pc1_centroids
        pc2 += pc2_centroids
        net1 = self.conv1(pc1, edge_index1, pc2, edge_index2)
        # print("net2")
        net2 = self.conv1(pc1, edge_index1, pc2, edge_index2)

        # print("knn")
        edge_index = knn_graph(net1, k=self.k, batch=pc1_batch)
        # print("net3")
        # print("edges: ", edge_index.shape )
        # print("net1: ", net1.shape )
        net3 = self.conv2(net1, edge_index)

        # print("nets")

        # print("net1 ",net1.shape)
        # print("net2 ",net2.shape)
        # print("net3 ",net3.shape)
        nets = torch.cat([net1, net2, net3], dim=-1)
        # print("catted", nets.shape)
        # print("out7")
        out7 = self.conv3(nets)
        # print("ou7: ", out7.shape)

        # print("max pool")
        global_features = global_max_pool(out7, pc1_batch)
        # print("global feature ", global_features.shape)
        expand_x = global_features[pc1_batch]
        #expand_x = x.repeat(1,num_points, 1, 1)

        # print("expand ", expand_x.shape)
        # print("net1 ",net1.shape)
        # print("net2 ",net2.shape)
        # print("net3 ",net3.shape)
        concat = torch.cat([expand_x, net1, net2, net3], dim=-1) # dim = 1024 + 64 + 64 + 256 = 1408

        # print("conv4")
        x = self.conv4(concat)
        # print("x: ", x.shape)
        # print("conv5")
        x = self.conv5(x)
        # print("x: ", x.shape)
        # print("conv6")
        x = self.conv6(x)
        # print("x: ", x.shape)
        # print("conv7")
        x = self.conv7(x)
        # print("x: ", x.shape)
        # print("donezo")

        # print("pc1: ", pc1.shape)
        # print("x: ", x.shape)
        out = pc1 + x
        out[..., :3] += centroids

        return out


class DGCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(DGCNN, self).__init__()

        nn = Seq(Lin(6, 64), ReLU(), Lin(64, 64), ReLU(), Lin(64, 64), ReLU())
        self.conv1 = EdgeConv(nn, aggr='max')

        nn = Seq(
            Lin(128, 128), ReLU(), Lin(128, 128), ReLU(), Lin(128, 256),
            ReLU())
        self.conv2 = EdgeConv(nn, aggr='max')

        self.lin0 = Lin(256, 512)

        self.lin1 = Lin(512, 256)
        self.lin2 = Lin(256, 256)
        self.lin3 = Lin(256, num_classes)

    def forward(self, pos, batch):
        edge_index = knn_graph(pos, k=20, batch=batch)
        x = self.conv1(pos, edge_index)

        edge_index = knn_graph(x, k=20, batch=batch)
        x = self.conv2(x, edge_index)

        x = F.relu(self.lin0(x))

        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)



class PointNet(BaseModel):
    def __init__(self):
        super(PointNet, self).__init__()

        self.local_sa1 = PointConv(
            Seq(Lin(3, 64), ReLU(), Lin(64, 64), ReLU(), Lin(64, 128)))

        self.local_sa2 = PointConv(
            Seq(Lin(131, 128), ReLU(), Lin(128, 128), ReLU(), Lin(128, 256)))

        self.global_sa = Seq(
            Lin(259, 256), ReLU(), Lin(256, 512), ReLU(), Lin(512, 1024))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 10)

    def forward(self, data):
        pos, batch = data.pos, data.batch

        idx = fps(pos, batch, ratio=0.5)  # 512 points
        edge_index = radius(pos[idx], pos, 0.1, batch[idx], batch, 48)
        x = F.relu(self.local_sa1(None, pos, edge_index))
        pos, batch = pos[idx], batch[idx]

        idx = fps(pos, batch, ratio=0.25)  # 128 points
        edge_index = radius(pos[idx], pos, 0.2, batch[idx], batch, 48)
        x = F.relu(self.local_sa2(x, pos, edge_index))
        pos, batch = pos[idx], batch[idx]

        x = self.global_sa(torch.cat([x, pos], dim=1))
        x = x.view(-1, 128, 1024).max(dim=1)[0]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)
