import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ModelNet


class MyModelNetDataLoader(DataLoader):
    """
    MNIST datasets loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, num_points, training=True):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        self.data_dir = data_dir
        path = osp.join(self.data_dir, 'ModelNet10')
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(num_points)

        train_dataset = ModelNet(path, '10', training, transform, pre_transform)

        super(MyModelNetDataLoader, self).__init__(train_dataset, batch_size=batch_size, shuffle=shuffle)


