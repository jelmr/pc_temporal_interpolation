import numpy as np
from scipy.special import softmax
from sklearn import preprocessing
import pandas as pd
import copy
import open3d
import os.path as osp
import sys
import argparse
import glob
from dynamic_point_cloud import FileBackedDynamicPointCloud


def upscale(pc, pcd, pcd_tree):
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)
    normals = np.asarray(pc.normals)
    dnormals = np.asarray(pcd.normals)
    n = -1
    for (point, color, normal) in zip(points[:n], colors[:n], normals[:n]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 10)
        knn = points[idx, :]

        knn_rel = knn - point

        knn_distance = np.linalg.norm(knn_rel, axis=1)

        max_ = np.max(knn_distance)

        reverse_knn = knn_distance*-1 + max_

        normalized_knn = np.expand_dims(reverse_knn / np.sum(reverse_knn), axis=-1)
        ncols = dnormals[idx, :] * normalized_knn
        ncols = ncols.sum(axis=0)
        normal[:] = ncols

    pc.normals = open3d.Vector3dVector(normals)
    return pc


def snap(dpcd, dpco):
    for j in range(len(dpcd)-1):
        print("Frame ",j)
    #     pc1 = copy.deepcopy(dpcd[j])
    #     pc2 = copy.deepcopy(dpcd[j+1])
        pc1 = dpcd[j]
        pc2 = dpcd[j+1]
        pc1_dest = np.asarray(pc1.points)[...] + np.asarray(pc1.normals)
        kdt = open3d.KDTreeFlann(pc2)

        p1 = np.asarray(pc1.points)
        c1 = np.asarray(pc1.colors)
        n1 = np.asarray(pc1.normals)
        n2 = np.asarray(pc2.normals)
        n = -1

        nn = pc1_dest.copy()
        p2_needsmap = np.ones((pc1_dest.shape[0]), dtype=bool)

        for i, (p1_, c1_, n1_) in enumerate(zip(pc1_dest[:n], c1[:n], n1[:n])):
            [k, idx, _] = kdt.search_knn_vector_3d(p1_, 1) #replace by hybrid
            nn[i, ...] = np.asarray(pc2.points)[idx, ...]
            p2_needsmap[idx] = False

        np.asarray(pc1.normals)[...] = nn - np.asarray(pc1.points)

        # Correct non attached p2s
        p2p = np.asarray(pc2.points)
        p2n = np.asarray(pc2.normals)
        p2_fix = np.zeros((p2_needsmap[p2_needsmap].shape[0], 3))
        for i, p_ in enumerate(p2p[p2_needsmap]):
            [k, idx, _] = kdt.search_knn_vector_3d(p_, 2) #replace by hybrid
            p2_fix[i, ...] = p2p[idx[1], ...]


        p2n[p2_needsmap, ...] = ( p2p[p2_needsmap,...] + p2n[p2_needsmap, ...] - p2_fix)
        p2p[p2_needsmap, ...] = p2_fix


        dpco.add_open3d_pc(naming.format(j), pc1)

    dpco.add_open3d_pc(naming.format(j+1), pc2)
    dpco.write_to_disk()



def upscale_dpc(dpc, dpcd, dpco):
    for i, (pc, pcd) in enumerate(zip(dpc, dpcd)):
        pc.normals = open3d.Vector3dVector(np.zeros(np.asarray(pc.points).shape))
        print("Frame {}".format(i))
        pcd_tree = open3d.KDTreeFlann(pcd)
        pcu = upscale(pc, pcd, pcd_tree)
        dpco.add_open3d_pc(osp.join(args.out_dir, "frame_{0:06d}.ply".format(i)), pcu)
    dpco.write_to_disk()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_hr_dir",
                        help=f"Base directory of input.",
                        type=str)
    parser.add_argument("in_hr_dpc",
                        help=f"Base directory of input.",
                        type=str)
    parser.add_argument("in_lr_dir",
                        help=f"Base directory of input.",
                        type=str)
    parser.add_argument("in_lr_dpc",
                        help=f"Base directory of input.",
                        type=str)
    parser.add_argument("out_dir",
                        help=f"Output directory [default=<same as input_dir>]",
                        type=str)
    parser.add_argument("--ascii",
                        help=f"Writes the output PLY file as ASCII [default={DEFAULT_ASCII}]",
                        action="store_true",
                        default=DEFAULT_ASCII)
    parser.add_argument("--remove", "-r",
                        help=f"Removes the texures, .obj and .mtl file. Also removes <input_dir> if <output_dir> is "
                        f"specified.",
                        action="store_true",
                        default=DEFAULT_REMOVE)
    parser.add_argument("--nosf",
                        help=f"Don't output any sceneflow.",
                        action="store_true",
                        default=False)
    parser.add_argument("--n", "-n",
                        type=int,
                        help=f"Number of points to sample from the mesh [default={DEFAULT_NUM_POINTS}]",
                        default=DEFAULT_NUM_POINTS)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    # Ensure output_dir exists
    pathlib.Path(osp.join(args.out_dir, "up")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(osp.join(args.out_dir, "snap")).mkdir(parents=True, exist_ok=True)

    dpc = FileBackedDynamicPointCloud(args.in_hr_dir, args.in_hr_frames)
    dpcd = FileBackedDynamicPointCloud(args.in_lr_dir, args.in_lr_frames)
    dpco = FileBackedDynamicPointCloud(osp.join(args.out_dir, "up"),osp.join(args.out_dir, "up", "frames.dpc"))
    dpco2 = FileBackedDynamicPointCloud(osp.join(args.out_dir, "snap"),osp.join(args.out_dir, "snap", "frames.dpc"))

    upscale_dpc(dpc, dpcd, dpco)
    snap(dpco, dpco2)
