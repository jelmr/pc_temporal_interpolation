import os.path as osp
import pickle
import sys
from itertools import product
#import open3d
from data.dynamic_point_cloud import FileBackedDynamicPointCloud
from utils.open3d_util import open3d_to_np_pc
import torch
import numpy as np
import torch_geometric.transforms as T
from transforms import (UniformSample,
                        NormalizeScale,
                        Jitter,
                        RandomScale,
                        RandomReverseFrames,
                        RandomFlip,
                        RandomRotate,
                        Shuffle)
import torch_geometric.data
from torch_geometric.data import Data, InMemoryDataset
#from utils.open3d_util import np_to_open3d_pc


class SyntheticAdvancedDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SyntheticAdvancedDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            "Jasper_SambaDancing",
        ]

    @property
    def processed_file_names(self):
        return ['data_0.pt']

    def download(self):
        raise NotImplemented("Downloading for this dataset is not supported.")


    def process_tf(self):
        points1 = []
        points2 = []
        points3 = []
        color1 = []
        color2 = []
        color3 = []
        normals1 = []

        data_list = []
        for i, raw_path in enumerate(self.raw_paths):
            print(f"[{i:03d}/{len(self.raw_paths):03d}] Processing {raw_path}")
            # Read data from `raw_path`.
            dpc = FileBackedDynamicPointCloud(raw_path, osp.join(raw_path, "frames.dpc"))
            frame_minus_2 = None
            frame_minus_1 = None
            for i, open3d_pc in enumerate(dpc):
                np_pc = open3d_to_np_pc(open3d_pc, normals=False)
                pt_pc = torch.FloatTensor(np_pc)

                if frame_minus_1 is None:
                    frame_minus_1 = pt_pc
                    continue
                elif frame_minus_2 is None:
                    frame_minus_2 = frame_minus_1
                    frame_minus_1 = pt_pc
                    continue

                pc1 = frame_minus_2.clone()
                num_points_pc1 = pc1.size()[0]
                pc2 = pt_pc.clone()
                num_points_pc2 = pc2.size()[0]
                target = frame_minus_1.clone()

                pcs = torch.cat([pc1, pc2], dim=-2)

                graph_id = torch.zeros(num_points_pc1 + num_points_pc2)
                graph_id[num_points_pc1:] = 1

                data = Data(pos=pcs, y=target)
                data.graph_id = graph_id

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                frame_minus_2 = frame_minus_1.clone()
                frame_minus_1 = pt_pc.clone()

                data_list.append(data)

                points1.append(data.pos[data.graph_id == 0].data.numpy()[..., :3])
                points2.append(data.y.data.numpy()[..., :3])
                points3.append(data.pos[data.graph_id == 1].data.numpy()[..., :3])

                color1.append(data.pos[data.graph_id == 0].data.numpy()[..., 3:6])
                color2.append(data.y.data.numpy()[..., 3:6])
                color3.append(data.pos[data.graph_id == 1].data.numpy()[..., 3:6])

                normals1.append(data.pos[data.graph_id == 0].data.numpy()[..., 6:])


        mask = np.zeros((len(points1), points1[0].shape[0]))
        d = {
            "points1":np.asarray(points1),
            "points2":np.asarray(points2),
            "points3":np.asarray(points3),
            "color1":np.asarray(color1),
            "color2":np.asarray(color2),
            "color3":np.asarray(color3),
            "normals1":np.asarray(normals1),
            "mask":np.asarray(mask)
        }

        print(self.processed_dir)
        with open(osp.join(self.processed_dir, 'training_data.pickle'), 'wb') as handle:
            pickle.dump(d, handle)

    def process(self):
        self.process_tf()
        print("Preprocessing complete.")
        sys.exit()

        i = 0


        data_list = []
        for i, raw_path in enumerate(self.raw_paths):
            print(f"[{i:03d}/{len(self.raw_paths):03d}] Processing {raw_path}")
            # Read data from `raw_path`.
            dpc = FileBackedDynamicPointCloud(raw_path, osp.join(raw_path, "frames.dpc"))
            frame_minus_2 = None
            frame_minus_1 = None
            for open3d_pc in dpc:
                np_pc = open3d_to_np_pc(open3d_pc)
                pt_pc = torch.FloatTensor(np_pc)

                if frame_minus_1 is None:
                    frame_minus_1 = pt_pc
                    continue
                elif frame_minus_2 is None:
                    frame_minus_2 = frame_minus_1
                    frame_minus_1 = pt_pc
                    continue

                # print(pt_pc.mean(dim=0))
                # pc1_open3d = np_to_open3d_pc(frame_minus_2.data.numpy())
                # pc1_open3d.paint_uniform_color([1,0.706,0])
                #
                # target_open3d = np_to_open3d_pc(frame_minus_1.data.numpy())
                # target_open3d.paint_uniform_color([0.706,1, 0])
                #
                # pc2_open3d = open3d_pc
                # pc2_open3d.paint_uniform_color([1, 0, 0.706])
                # open3d.draw_geometries([pc1_open3d, target_open3d, pc2_open3d])

                # print("=======================================")
                # print("NP PC size: ", np_pc.shape)
                # print("PC size: ", pt_pc.size())
                # print("=======================================")

                pc1 = frame_minus_2.clone()
                num_points_pc1 = pc1.size()[0]
                pc2 = pt_pc.clone()
                num_points_pc2 = pc2.size()[0]
                target = frame_minus_1.clone()

                pcs = torch.cat([pc1, pc2], dim=-2)

                graph_id = torch.zeros(num_points_pc1 + num_points_pc2)
                graph_id[num_points_pc1:] = 1

                data = Data(pos=pcs, y=target)
                data.graph_id = graph_id
                #data.num_points_pc1 = torch.IntTensor(num_points_pc1)
                #data.num_points_pc2 = torch.IntTensor(num_points_pc2)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                # TODO: Transforms should be done before triple generation
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                frame_minus_2 = frame_minus_1.clone()
                frame_minus_1 = pt_pc.clone()

                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), osp.join(self.processed_dir, 'data_{}.pt'.format(0)))


class SyntheticAdvancedDataLoader(torch_geometric.data.DataLoader):
    """
        TODO: Doc
    """

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, num_points, training=True):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        self.data_dir = data_dir
        path = osp.join(self.data_dir, 'SyntheticAdvanced')
        pre_transform = T.Compose([
            UniformSample(num_points),
            NormalizeScale()
        ])

        transform = T.Compose([
            RandomFlip(p=0.5, flip_x=True, flip_y=False, flip_z=True),
            RandomReverseFrames(p=0.5),
            RandomRotate(degree_range=(-15, 15), axis=0),
            RandomRotate(degree_range=(0, 360), axis=1),
            RandomRotate(degree_range=(-15, 15), axis=2),
            RandomScale(scales=(0.9, 1.1)),
            Jitter(jitter_range=0.0002, uniform=True, clip=(torch.tensor([
                [-float("inf"), -float("inf"), -float("inf"), 0, 0, 0],
                [float("inf"), float("inf"), float("inf"), 1, 1, 1],
            ]))),
            Shuffle()
        ])

        train_dataset = SyntheticAdvancedDataset(path, transform, pre_transform)
        # train_dataset = ModelNet(path, '10', training, transform, pre_transform)

        super(SyntheticAdvancedDataLoader, self).__init__(train_dataset, batch_size=batch_size, shuffle=shuffle)
