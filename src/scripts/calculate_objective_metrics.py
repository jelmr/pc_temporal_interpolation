import pickle
import math
import numpy as np
import scipy
import scipy.ndimage
import open3d
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os.path as osp
import argparse
import os
import copy
import sys
from dynamic_point_cloud import FileBackedDynamicPointCloud as DPC

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir",
                        help=f"Base directory of input.",
                        type=str)
    parser.add_argument("gt_dir",
                        help=f"Base directory of input.",
                        type=str)
    parser.add_argument("out",
                        help=f"Base directory of input.",
                        type=str)
    return parser.parse_args()



def to_np(pc, normals=True):
    num_points = len(pc.points)

    if normals:
        pc_np = np.zeros((num_points, 9))
    else:
        pc_np = np.zeros((num_points, 6))

    pc_np[..., :3] = np.asarray(pc.points)
    pc_np[..., 3:6] = np.asarray(pc.colors)

    if normals:
        pc_np[..., 6:] = np.asarray(pc.normals)


    return pc_np


def to_3d(pc):
    pc_open3d = open3d.PointCloud()

    xyzs = open3d.Vector3dVector(pc[..., :3])
    cols = open3d.Vector3dVector(pc[..., 3:])

    pc_open3d.points = xyzs
    pc_open3d.colors = cols

    return pc_open3d

def p2p_one_way_np(pc1, pc2):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # Ensure inputs are the same shape
#     pc1 = to_np(dpci[0])
#     pc2 = to_np(dpci[0])

    kdt = open3d.KDTreeFlann(pc2)

    p1 = np.asarray(pc1.points)
    c1 = np.asarray(pc1.colors)
    n1 = np.asarray(pc1.normals)
    n2 = np.asarray(pc2.normals)
    n = -1

    nn_xyz = np.asarray(pc1.points).copy()
    nn_rgb = np.asarray(pc1.colors).copy()

    for i, (p1_, c1_, n1_) in enumerate(zip(p1[:n], c1[:n], n1[:n])):
        [k, idx, _] = kdt.search_knn_vector_3d(p1_, 1) #replace by hybrid
        nn_xyz[i, ...] = np.asarray(pc2.points)[idx, ...]
        nn_rgb[i, ...] = np.asarray(pc2.colors)[idx, ...]



    # For each pair of points, calculate the squared error.

    xyz_dist = np.square(nn_xyz - p1)
    xyz_dist = np.sum(xyz_dist, axis=-1, keepdims=True)

    rgb_dist = np.square(nn_rgb - c1)
    rgb_dist = np.mean(rgb_dist, axis=-1, keepdims=True)

    xyz_min = np.min(p1[..., :3], axis=0)
    xyz_max = np.max(p1[..., :3], axis=0)
    xyz_range = xyz_max - xyz_min

    bounding_box_width = np.max(xyz_range)


    err_xyz = xyz_dist / (3 * (bounding_box_width**2))
    err_rgb = rgb_dist / (255 ** 2)



    xyz_mean = np.mean(err_xyz)
    rgb_mean = np.mean(err_rgb)
    return xyz_mean, rgb_mean



def p2p_symmetric(pc1, pc2, normal):
    xyz_1, rgb_1 = p2p_one_way_np(pc1, pc2)
    xyz_2, rgb_2 = p2p_one_way_np(pc2, pc1)

    xyz = max(xyz_1, xyz_2) + 1e-20
    psnr_xyz = -10 * np.log10(xyz)

    rgb = max(rgb_1, rgb_2) + 1e-20
    psnr_rgb = -10 * np.log10(rgb)

    return psnr_xyz, psnr_rgb



def p2plane_np(pc1, pc2, normal):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # Ensure inputs are the same shape
    pc1p = np.asarray(pc1.points)
    pc2p = np.asarray(pc2.points)

    num_points_1 = pc1p.shape[1]
    num_points_2 = pc2p.shape[1]

    pc1p = np.expand_dims(pc1p, 1)
    pc2p = np.expand_dims(pc2p, 1)


    kdt = open3d.KDTreeFlann(pc2)

    p1 = np.asarray(pc1.points)
    c1 = np.asarray(pc1.colors)
    n1 = np.asarray(pc1.normals)
    n2 = np.asarray(pc2.normals)
    n = -1

    nn_xyz = np.asarray(pc1.points).copy()
    nn_rgb = np.asarray(pc1.colors).copy()

    for i, (p1_, c1_, n1_) in enumerate(zip(p1[:n], c1[:n], n1[:n])):
        [k, idx, _] = kdt.search_knn_vector_3d(p1_, 1) #replace by hybrid
        nn_xyz[i, ...] = np.asarray(pc2.points)[idx, ...]
        nn_rgb[i, ...] = np.asarray(pc2.colors)[idx, ...]


    # For each pair of points, calculate the squared error.
    xyz_dist = np.square(nn_xyz - p1)
    error = np.reshape(xyz_dist, (-1, 1, 3))
    normal = np.reshape(normal, (-1, 3, 1))

    projected_error = np.square(np.matmul(error, normal))
    projected_error = np.reshape(projected_error, (-1 ))

    xyz_min = np.min(p1[..., :3], axis=0)
    xyz_max = np.max(p1[..., :3], axis=0)
    xyz_range = xyz_max - xyz_min
    bounding_box_width = np.max(xyz_range)
    xyz_err = np.mean(projected_error / (3 * (bounding_box_width**2))) + 1e-20
    psnr_xyz = -10 * np.log10(xyz_err)
    return [psnr_xyz.item()]





def vifp_measure(ref, dist):

    sigma_nsq=2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2**(4-scale+1) + 1
        sd = N/5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq<eps] = 0
        sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
        sigma1_sq[sigma1_sq<eps] = 0

        g[sigma2_sq<eps] = 0
        sv_sq[sigma2_sq<eps] = 0

        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num/den

    return vifp


def get_projections(pc1, resolution_factor):
    # TODO: remove redundancy
    # TODO: Take pandas out
    r = np.around(pc1[..., :3] * resolution_factor, decimals=0).astype(int)
    r -= np.min(r, axis=0)

    # proj 1
    groups = pd.DataFrame(r).groupby([0,1])
    idx = groups.idxmax().values
    shape = np.max(r[idx, ...], axis=0).astype(int).squeeze()
    proj1 = np.zeros((shape[0]+1, shape[1]+1, 3))
    zeros_idx = r[idx, :2].astype(int).squeeze()
    proj1[zeros_idx[..., 0], zeros_idx[..., 1]] = pc1[idx, 3:].squeeze()

    # proj2
    idx = groups.idxmin().values
    shape = np.max(r[idx, ...], axis=0).astype(int).squeeze()
    proj2 = np.zeros((shape[0]+1, shape[1]+1, 3))
    zeros_idx = r[idx, :2].astype(int).squeeze()
    proj2[zeros_idx[..., 0], zeros_idx[..., 1]] = pc1[idx, 3:].squeeze()

    #proj 3
    groups = pd.DataFrame(r).groupby([1,2])
    idx = groups.idxmin().values
    shape = np.max(r[idx, ...], axis=0).astype(int).squeeze()
    proj3 = np.zeros((shape[1]+1, shape[2]+1, 3))
    zeros_idx = r[idx, 1:].astype(int).squeeze()
    proj3[zeros_idx[..., 0], zeros_idx[..., 1]] = pc1[idx, 3:].squeeze()

    #proj 4
    idx = groups.idxmin().values
    shape = np.max(r[idx, ...], axis=0).astype(int).squeeze()
    proj4 = np.zeros((shape[1]+1, shape[2]+1, 3))
    zeros_idx = r[idx, 1:].astype(int).squeeze()
    proj4[zeros_idx[..., 0], zeros_idx[..., 1]] = pc1[idx, 3:].squeeze()

    #proj 5
    groups = pd.DataFrame(r).groupby([0,2])
    idx = groups.idxmax().values
    shape = np.max(r[idx, ...], axis=0).astype(int).squeeze()
    proj5 = np.zeros((shape[0]+1, shape[2]+1, 3))
    zeros_idx = r[idx, [0,2]].astype(int).squeeze()
    proj5[zeros_idx[..., 0], zeros_idx[..., 1]] = pc1[idx, 3:].squeeze()

    #proj 6
    idx = groups.idxmin().values
    shape = np.max(r[idx, ...], axis=0).astype(int).squeeze()
    proj6 = np.zeros((shape[0]+1, shape[2]+1, 3))
    zeros_idx = r[idx, [0,2]].astype(int).squeeze()
    proj6[zeros_idx[..., 0], zeros_idx[..., 1]] = pc1[idx, 3:].squeeze()

    return proj1, proj2, proj3, proj4, proj5, proj6


def pad(p, q):
    x = max(p.shape[0], q.shape[0])
    y = max(p.shape[1], q.shape[1])
    p_ = np.zeros((x,y,3))
    p_[:p.shape[0], :p.shape[1], :3] = p
    q_ = np.zeros((x,y,3))
    q_[:q.shape[0], :q.shape[1], :3] = q
    return p_, q_


def plot_projections(proj1, proj2, proj3, proj4, proj5, proj6):
    fig = plt.figure()
    fig, ax = plt.subplots(2, 3)
    ax[0,0].imshow(np.rot90(proj1))
    ax[1,0].imshow(np.rot90(proj2))
    ax[0,1].imshow(proj3)
    ax[1,1].imshow(proj4)
    ax[0,2].imshow(proj5)
    ax[1,2].imshow(proj6)
    fig.set_figheight(15)
    fig.set_figwidth(15)


def projected_vifp(pc1, pc2, normals, r=24, plot=False):
    pc1 = to_np(pc1, normals=False)
    pc2 = to_np(pc2, normals=False)
    p1,p2,p3,p4,p5,p6 = get_projections(pc1, r)
    q1,q2,q3,q4,q5,q6 = get_projections(pc2, r)

    if plot:
        plot_projections(p1,p2,p3,p4,p5,p6)
        plot_projections(q1,q2,q3,q4,q5,q6)

    vifp = (vifp_measure(*pad(p1, q1)) +         vifp_measure(*pad(p2, q2)) +         vifp_measure(*pad(p3, q3)) +         vifp_measure(*pad(p4, q4)) +         vifp_measure(*pad(p5, q5)) +         vifp_measure(*pad(p6, q6))) / 6
    print(vifp)
    return [vifp]

def scene_flow_EPE_np(pred, labels, normals):
    global CM

    ACC1_REL = 0
    ACC1_ABS = 0.5 * CM
    ACC2_REL = 0
    ACC2_ABS = 1 * CM
    ACC3_REL = 0
    ACC3_ABS = 2 * CM
    pred = np.asarray(pred.normals)
    labels = np.asarray(labels.normals)
    error = np.sqrt(np.sum((pred - labels)**2, axis=-1) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, axis=-1) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= ACC1_ABS), (error/gtflow_len <= ACC1_REL)), axis=-1)
    acc1 /= pred.shape[0]
    acc2 = np.sum(np.logical_or((error <= ACC2_ABS), (error/gtflow_len <= ACC2_REL)), axis=-1)
    acc2 /= pred.shape[0]
    acc3 = np.sum(np.logical_or((error <= ACC3_ABS), (error/gtflow_len <= ACC3_REL)), axis=-1)
    acc3 /= pred.shape[0]
    EPE = np.sum(error, axis=-1)/ error.shape[0] / CM
    EPE = np.mean(EPE)

    return EPE, acc1, acc2, acc3



def eval_metric(metric_fn, dpci, dpcgt, dpcgt_nm, names, result,interpolation_factor=4):
    len_i = len(dpci)
    len_gt = len(dpcgt)


    count = 0
    total = 0
    for i in range(len_i):
        print(i)
        pci = dpci[i]
        pcgt = dpcgt[i]
        pcgt_nm = dpcgt_nm[i]

        pci_p = np.asarray(pci.points)
        pci_n = np.asarray(pci.normals)
        pcgt_p = np.asarray(pcgt.points)
        pcgt_n = np.asarray(pcgt.normals)

        normals = np.asarray(pcgt_nm.normals)

        for i in range(interpolation_factor):
            pci_p += pci_n / interpolation_factor
            pcgt_p += pcgt_n / interpolation_factor


            x = list(metric_fn(pci, pcgt, normals))
            for i, name in enumerate(names):
                result[name].append(x[i])


    return result


def eval_all_metrics(dpci, dpcgt, dpcgt_nm):
    global CM

    p = np.asarray(dpcgt[0].points)
    x_min = np.min(p, axis=0)[1]
    x_max = np.max(p, axis=0)[1]
    x_range = x_max - x_min
    CM = x_range / 175

    result = defaultdict(list)
    result = eval_metric(p2p_symmetric, copy.deepcopy(dpci), copy.deepcopy(dpcgt), copy.deepcopy(dpcgt_nm), result=result, names=["PSNR_XYZ", "PSNR_RGB"])
    result = eval_metric(p2plane_np, copy.deepcopy(dpci), copy.deepcopy(dpcgt), copy.deepcopy(dpcgt_nm), result=result, names=["PSNR_P2PLANE"])
    result = eval_metric(projected_vifp, dpci, dpcgt, dpcgt_nm, result=result, names=["PROJ_VIFP"])
    result = eval_metric(scene_flow_EPE_np, dpci, dpcgt, dpcgt_nm, result=result, names=["EPE", "ACC_05", "ACC_10", "ACC_20"])
    print(result)
    means = {}
    for k in result:
        means[k] = np.mean(list(filter(math.isfinite, result[k])))
    return means


if __name__ == "__main__":
    args = parse_args()

    dpci_dir = args.in_dir
    dpci = DPC(dpci_dir, osp.join(dpci_dir, "frames.dpc"))


    dpcgt_sf_dir = osp.join(args.gt_dir, "sceneflow")
    dpcgt_sf = DPC(dpcgt_sf_dir, osp.join(dpcgt_sf_dir, "frames.dpc"))

    dpcgt_nm_dir = osp.join(args.gt_dir, "normals")
    dpcgt_nm = DPC(dpcgt_nm_dir, osp.join(dpcgt_nm_dir, "frames.dpc"))

    print("Input DPC: ", dpci_dir)
    print("GT (sf): ", dpcgt_sf_dir)
    print("GT (nm): ", dpcgt_nm_dir)

    means = eval_all_metrics(dpci, dpcgt_sf, dpcgt_nm)

    with open(args.out, 'wb') as handle:
        pickle.dump(means, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for k in means:
        print(k, ": ", means[k])

