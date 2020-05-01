import open3d
import numpy as np


def np_to_open3d_pc(pc):
    pc_open3d = open3d.PointCloud()

    xyzs = open3d.Vector3dVector(pc[..., :3])
    cols = open3d.Vector3dVector(pc[..., 3:])

    pc_open3d.points = xyzs
    pc_open3d.colors = cols

    return pc_open3d


def open3d_to_np_pc(pc, normals=False):
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
