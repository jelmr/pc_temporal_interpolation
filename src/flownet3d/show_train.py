'''
    Single-GPU training code
'''

import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
#import synthetic_dataset
import rigid_dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


NUM_POINT=1024
BATCH_SIZE = 8
#DATA = "/home/jelmer/data/with_normals/data_0.pickle"
DATA = "/home/jelmer/data/scene.pickle"
#TRAIN_DATASET = synthetic_dataset.SceneflowDataset(DATA, npoints=NUM_POINT)
#TEST_DATASET = synthetic_dataset.SceneflowDataset(DATA, npoints=NUM_POINT, train=False)
TRAIN_DATASET = rigid_dataset.SceneflowDataset(npoints=NUM_POINT)
TEST_DATASET = rigid_dataset.SceneflowDataset(npoints=NUM_POINT, train=False)



def train():

        train_one_epoch()


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT*2, 6))
    batch_label = np.zeros((bsize, NUM_POINT, 6))
    batch_mask = np.zeros((bsize, NUM_POINT))
    batch_normals = np.zeros((bsize, NUM_POINT, 3))
    # shuffle idx to change point order (change FPS behavior)
    shuffle_idx = np.arange(NUM_POINT)
    np.random.shuffle(shuffle_idx)
    for i in range(bsize):
        pc1, pc2, pc3, color1, color2, color3, normals1, mask1 = dataset[idxs[i+start_idx]]
        # move pc1 to center
        pc1_center = np.mean(pc1, 0)
        pc1 -= pc1_center
        pc3 -= pc1_center
        batch_data[i,:NUM_POINT,:3] = pc1[shuffle_idx, ...]
        batch_data[i,:NUM_POINT,3:] = color1[shuffle_idx, ...]
        batch_data[i,NUM_POINT:,:3] = pc2[shuffle_idx, ...]
        batch_data[i,NUM_POINT:,3:] = color2[shuffle_idx, ...]
        batch_label[i, :NUM_POINT, :3] = pc2[shuffle_idx, ...]
        batch_label[i, :NUM_POINT, 3:] = color2[shuffle_idx, ...]
        batch_mask[i] = mask1[shuffle_idx, ...]
        batch_normals[i, :NUM_POINT, :3] = normals1[shuffle_idx, ...]
    return batch_data, batch_label, batch_mask, batch_normals

def train_one_epoch():
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET) // BATCH_SIZE

    from utils.open3d_util import np_to_open3d_pc
    import open3d
    vis = open3d.Visualizer()
    vis.create_window()

    pc1_ = open3d.geometry.PointCloud()
    pc2_ = open3d.geometry.PointCloud()
    b=False
    o = vis.get_render_option()
    o.point_show_normal = True
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_mask, batch_normals = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)

        for i in range(BATCH_SIZE):
            pc1 = batch_data[i, :NUM_POINT, :]
            pc2 = batch_data[i, NUM_POINT:, :]

            pc1_.normals = open3d.Vector3dVector(batch_normals[i, ...])
            pc1_.points = open3d.Vector3dVector(batch_data[i, :NUM_POINT, :3])
            pc1_.colors = open3d.Vector3dVector(batch_data[i, :NUM_POINT, 3:])

            #np.asarray(pc1_.points)[...] += np.asarray(pc1_.normals) + 0.001

            pc2_.points = open3d.Vector3dVector(batch_data[i, NUM_POINT:, :3])
            pc2_.colors = open3d.Vector3dVector(batch_data[i, NUM_POINT:, 3:])

            pc1_.normals = open3d.Vector3dVector([])
            pc2_.normals = open3d.Vector3dVector([])

            if not b:
                vis.add_geometry(pc1_)
                vis.add_geometry(pc2_)
                b=True

            pc1_.paint_uniform_color([1, 0.706, 0])
            pc2_.paint_uniform_color([0.706, 0, 1])
            #open3d.draw_geometries([pc1, pc2])
            vis.update_geometry()
            vis.update_renderer()
            vis.poll_events()
            vis.run()
            input()

    vis.destroy_window()




if __name__ == "__main__":
    train()
