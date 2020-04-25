'''
TODO: This was quickly thrown together from old implementation. Direly needs refactoring!!
'''


import os.path as osp
from itertools import product
#import open3d
import os
import random

import h5py as h5py
import numpy as np
from data.dynamic_point_cloud import FileBackedDynamicPointCloud
from utils.open3d_util import open3d_to_np_pc
import torch
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
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
#from utils.open3d_util import np_to_open3d_pc

CIRCLE_RADIUS_RANGE = [0.5, 1.5]
FRAMES_PER_CIRCLE_RANGE = [50, 80]
HEIGHT_PER_CIRCLE_RANGE = [1., 2.]
FRAMES_PER_ROTATION_RANGE = [50, 80]
FRAMES_PER_SHRINK_RANGE = [50, 80]
SHRINK_RANGE = [0.3, 1.2]
FRAME_RANGE = [1, 320]


def get_video_style_pc_permutations(pc,
                                    circle_radius=None,
                                    frames_per_circle=None,
                                    height_per_circle=None,
                                    frames_per_rotation=None,
                                    shrink_range=None,
                                    frames_per_shrink=None):
    B, N, D = pc.shape

    res_1 = np.zeros((B, N, D))
    res_2 = np.zeros((B, N, D))
    res_3 = np.zeros((B, N, D))

    idx = list(range(B))
    # print(">>> B: ", B)
    random.shuffle(idx)

    # TODO: Ideally use NP vectorization here, though performance does not seem to be a huge issue for this part.

    if circle_radius is None:
        # print("Random circle radius.")
        circle_radius = random.uniform(*CIRCLE_RADIUS_RANGE)

    if frames_per_circle is None:
        # print("Random frames per circle.")
        frames_per_circle = random.randrange(*FRAMES_PER_CIRCLE_RANGE)

    if height_per_circle is None:
        # print("Random height per circle.")
        height_per_circle = random.uniform(*HEIGHT_PER_CIRCLE_RANGE)

    if frames_per_rotation is None:
        # print("Random frames per rotation.")
        frames_per_rotation_x = random.randrange(*FRAMES_PER_ROTATION_RANGE)
        frames_per_rotation_y = random.randrange(*FRAMES_PER_ROTATION_RANGE)
        frames_per_rotation_z = random.randrange(*FRAMES_PER_ROTATION_RANGE)
    else:
        frames_per_rotation_x = frames_per_rotation
        frames_per_rotation_y = frames_per_rotation
        frames_per_rotation_z = frames_per_rotation

    if shrink_range is None:
        # print("Random shrink range.")
        shrink_range = random.uniform(*SHRINK_RANGE)

    if frames_per_shrink is None:
        # print("Random frames per shrink.")
        frames_per_shrink = random.randrange(*FRAMES_PER_SHRINK_RANGE)

    for i in range(B):
        frame_index_shrink = random.randrange(*FRAME_RANGE)
        frame_index_shift = random.randrange(*FRAME_RANGE)
        frame_index_rotation = random.randrange(*FRAME_RANGE)

        # shifts = shift_straight(n_frames, np.asarray([0.3, 0.6, 0]))
        shifts_fn = shift_circle(circle_radius, frames_per_circle=frames_per_circle,
                                 height_per_circle=height_per_circle)

        # rotations = rotate_none(n_frames)
        rotations_fn = rotate(frames_per_rotation_x, frames_per_rotation_y, frames_per_rotation_z)
        shrinks_fn = shrink_sin(shrink_range, frames_per_shrink)

        # shrinks = shrink_sin(n_frames, 60)

        # shifts = mutations_from_function(shifts_fn, n_frames, dtype=np.float64)
        # rotations = mutations_from_function(rotations_fn, dtype=np.float64)
        # shrinks = mutations_from_function(shrinks_fn, dtype=np.float64)

        shifts = triplet_from_function(frame_index_shift, shifts_fn)
        rotations = triplet_from_function(frame_index_rotation, rotations_fn)
        shrinks = triplet_from_function(frame_index_shrink, shrinks_fn)
        # print("SHIFTS: ", shifts)
        # print("ROTATIONS: ", rotations)
        # print("SHRINKS: ", shrinks)

        pc1, pc2, pc3 = create_video_frames(pc[i, ...], 3, shifts, rotations, shrinks)
        res_1[idx[i], ...] = pc1
        res_2[idx[i], ...] = pc2
        res_3[idx[i], ...] = pc3

        # import visualizer
        # visualizer.plot_comparison(1, pc1, pc2, pc3, pc2)

    # return frames
    return res_1, res_2, res_3

def get_object_pcs(data, labels, names, object):
    object_index = np.argwhere(names == object).squeeze()
    labels = np.asarray(labels)
    object_indices = np.argwhere(labels == object_index)
    relevant_data = data[object_indices].squeeze()[:, 0]

    colored_data = add_color(relevant_data)

    return colored_data


def add_color(relevant_data):
    colored_data = np.tile(relevant_data, (1, 1, 2))
    colored_data[..., 3:] = colored_data[..., 3:] * 0.5 + 0.5
    return colored_data


def get_label_names(file):
    return np.asarray(open(file).read().splitlines())

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def create_pc(raw_pc):
    xyz = raw_pc
    rgb = xyz / 2 + 0.5
    return np.concatenate((xyz, rgb), axis=1)


def get_object_from_input(input_file, names_file, object=None):
    names = get_label_names(names_file)
    data, labels = load_h5(input_file)

    if object:
        return get_object_pcs(data, labels, names, object)
    else:
        return add_color(data)



def mutations_from_function(fn, n_frames, dtype=np.int64):
    return np.fromfunction(fn, (n_frames, 3), dtype=dtype)


def rotate_none():
    def fn(x, y):
        return x * 0

    return fn


def rotate(frames_per_x, frames_per_y, frames_per_z):
    # rotation_per_frame = 2 * np.pi / frames_per_rotation
    rotation_per_frame = np.zeros((3))
    if frames_per_x != 0:
        rotation_per_frame[0] = 2 * np.pi / frames_per_x
    if frames_per_y != 0:
        rotation_per_frame[0] = 2 * np.pi / frames_per_y
    if frames_per_z != 0:
        rotation_per_frame[0] = 2 * np.pi / frames_per_z

    def fn(x, y):
        return x * rotation_per_frame

    return fn


def shrink_none():
    def fn(x, y):
        return 1 + 0 * x

    return fn


def shrink_sin(shrink_range, frames_per_cycle):
    def fn(x, y):
        return 1.0 + shrink_range * abs(np.sin(x * 2 * np.pi / frames_per_cycle))

    return fn


def shift_straight(distance_per_frame):
    def fn(x, y):
        return x * np.take(distance_per_frame, y, axis=-1)

    return fn


def shift_circle(radius, frames_per_circle, height_per_circle):
    def fn(x, y):
        res = x
        # res[..., 0] = x[..., 0] * height_per_circle / frames_per_circle
        res[..., 0] = radius * np.sin(x[..., 1] * 1.5 * np.pi / frames_per_circle)
        res[..., 1] = radius * np.sin(x[..., 1] * 2 * np.pi / frames_per_circle)
        res[..., 2] = radius * np.cos(x[..., 2] * 2 * np.pi / frames_per_circle)

        return res

    return fn

def triplet_from_function(frame_index, shifts_fn):
    base_x = np.tile(np.arange(frame_index, frame_index + 3).reshape(3, 1), (1, 3)).astype(np.float64)
    base_y = np.tile(np.arange(0, 3).reshape(1, 3), (3, 1)).astype(np.float64)
    shifts = shifts_fn(base_x, base_y)
    return shifts

def rotate_point_cloud_by_angle(batch_data, rotation_angle=None, axis=0):
    """ Rotate the point cloud along up direction with certain angle.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        if rotation_angle is None:
            angle = np.random.uniform() * 2 * np.pi
        else:
            angle = rotation_angle[k, ...]
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, cosval, -sinval],
                                      [0, sinval, cosval]])
        rotation_matrix_y = np.array([[cosval, 0, sinval],
                                      [0, 1, 0],
                                      [-sinval, 0, cosval]])
        rotation_matrix_z = np.array([[cosval, -sinval, 0],
                                      [sinval, cosval, 0],
                                      [0, 0, 1]])
        shape_pc = batch_data[k, :, :3]

        if axis == 0:
            rotation_matrix = rotation_matrix_x
        elif axis == 1:
            rotation_matrix = rotation_matrix_y
        else:
            rotation_matrix = rotation_matrix_z

        rotated_data[k, :, :3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    rotated_data[:, :, 3:] = batch_data[:, :, 3:]
    return rotated_data


def create_video_frames(base_pc,
                        n_frames,
                        shifts,
                        rotations,
                        shrinks):
    pcs = np.tile(base_pc, (n_frames, 1, 1))

    # Shrink
    shrinks = shrinks.reshape([n_frames, 1, 3])
    centroids = np.expand_dims(np.mean(pcs[..., :3], axis=-2), axis=-2)
    pcs[..., :3] = pcs[..., :3] - centroids
    pcs[..., :3] = pcs[..., :3] * shrinks
    pcs[..., :3] = pcs[..., :3] + centroids

    # Rotate
    pcs = rotate_point_cloud_by_angle(pcs, axis=0, rotation_angle=rotations[..., 0])
    pcs = rotate_point_cloud_by_angle(pcs, axis=1, rotation_angle=rotations[..., 1])
    pcs = rotate_point_cloud_by_angle(pcs, axis=2, rotation_angle=rotations[..., 2])

    # Shift
    for frame_index in range(n_frames):
        pcs[frame_index, :, :3] += shifts[frame_index, :]

    return pcs
























class BasicTransform(object):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`.
    """

    def __init__(self):
        pass

    def __call__(self, data):
        # print("=========== CALL =============")
        # print("data: ", data)
        # print("data.pos: ", data.pos)
        # print("data.pos.shape: ", data.pos.shape)
        objs = data.pos.data.numpy()
        objs = np.expand_dims(objs, axis=0)
        # print(objs.shape)

        pc1, labels, pc2 = get_video_style_pc_permutations(objs)


        pc1 = torch.FloatTensor(pc1).squeeze()
        num_points_pc1 = pc1.size()[0]

        labels = torch.FloatTensor(labels).squeeze()

        pc2 = torch.FloatTensor(pc2).squeeze()
        num_points_pc2 = pc2.size()[0]

        pcs = torch.cat([pc1, pc2], dim=-2)

        graph_id = torch.zeros(num_points_pc1 + num_points_pc2)
        graph_id[num_points_pc1:] = 1

        data.pos = pcs
        data.y = labels
        data.graph_id = graph_id

        # print("pos: ", data.pos.size())
        # print("y: ", data.y.size())
        # print("graph_id: ", data.graph_id.size())
        # print(data.pos.type())
        # print(data.y.type())
        # print(data.graph_id.type())


        # print("=========== END OF CALL =============")
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class SyntheticBasicDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        # print("Root: ", root)
        super(SyntheticBasicDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    url = 'https://shapenet.cs.stanford.edu/media/'



    @property
    def raw_file_names(self):
        return ['modelnet40_ply_hdf5_2048']


    @property
    def processed_file_names(self):
        return ['data_0.pt']

    def download(self):
        for name in self.raw_file_names:
            url = '{}/{}.zip'.format(self.url, name)
            path = download_url(url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        i = 0

        input_file = f"{self.raw_paths[0]}/ply_data_train0.h5"
        names_file = f"{self.raw_paths[0]}/shape_names.txt"

        data_list = []

        objs = get_object_from_input(input_file, names_file)

        for i in range(objs.shape[0]):
            # print(".", end='')
            pc = objs[i, ...]
            pc = torch.tensor(pc)

            data = Data(pos=pc)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # print("Donezo, saving!")

        data, slices = self.collate(data_list)
        torch.save((data, slices), osp.join(self.processed_dir, 'data_{}.pt'.format(0)))






class SyntheticBasicDataLoader(torch_geometric.data.DataLoader):
    """
        TODO: Doc
    """

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, num_points, training=True):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        self.data_dir = data_dir
        path = osp.join(self.data_dir, 'SyntheticBasic')
        pre_transform = None

        transform = T.Compose([
            BasicTransform(),
            RandomFlip(p=0.5, flip_x=True, flip_y=False, flip_z=True),
            RandomReverseFrames(p=0.5),
            RandomRotate(degree_range=(-15, 15), axis=0),
            RandomRotate(degree_range=(0, 360), axis=1),
            RandomRotate(degree_range=(-15, 15), axis=2),
            #RandomScale(scales=(0.9, 1.1)),
            # Jitter(jitter_range=0.0002, uniform=True, clip=(torch.tensor([
            #     [-float("inf"), -float("inf"), -float("inf"), 0, 0, 0],
            #     [float("inf"), float("inf"), float("inf"), 1, 1, 1],
            # ]))),
            Shuffle()
        ])

        train_dataset = SyntheticBasicDataset(path, transform, pre_transform)
        # train_dataset = ModelNet(path, '10', training, transform, pre_transform)

        super(SyntheticBasicDataLoader, self).__init__(train_dataset, batch_size=batch_size, shuffle=shuffle)
