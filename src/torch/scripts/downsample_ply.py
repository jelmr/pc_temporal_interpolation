import argparse
import numpy as np
import open3d
import os.path as osp
import pathlib
import sys

BASE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), osp.pardir)
sys.path.append(BASE_DIR)
sys.path.append(osp.join(BASE_DIR, 'src'))

from data.dynamic_point_cloud import FileBackedDynamicPointCloud

DEFAULT_POINTS = 2048
DEFAULT_METHOD = "uniform"
DEFAULT_ASCII = True


def downsample_uniform(pc, points):
    input_points = np.asarray(pc.points)
    choice = np.random.choice(input_points.shape[0], points, replace=False)

    points = input_points[choice, :]
    colors = np.asarray(pc.colors)[choice, :]

    pc.points = open3d.Vector3dVector(points)
    pc.colors = open3d.Vector3dVector(colors)

    return pc


def downsample(pc, method, points):
    if method.lower() == "uniform":
        return downsample_uniform(pc, points)
    else:
        raise NotImplemented


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply",
                        help=f"Input PLY file.",
                        type=str)
    parser.add_argument("output_ply",
                        help=f"Output PLY file.",
                        type=str)
    parser.add_argument("--points",
                        help=f"Number of points to downsample to [default={DEFAULT_POINTS}]",
                        type=int,
                        default=DEFAULT_POINTS)
    parser.add_argument("--method",
                        help=f"Method to use to downsample [uniform, default={DEFAULT_METHOD}]",
                        type=str,
                        default=DEFAULT_METHOD)
    parser.add_argument("--ascii",
                        help=f"Writes the output PLY file as ASCII [default={DEFAULT_ASCII}]",
                        action="store_true",
                        default=DEFAULT_ASCII)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Read input PLY
    input_pc = open3d.read_point_cloud(args.input_ply)

    # Convert it
    output_pc = downsample(input_pc, args.method, args.points)

    # Create output dir if needed
    output_dir, output_file = osp.split(args.output_ply)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write output PLY
    open3d.write_point_cloud(args.output_ply, output_pc, write_ascii=args.ascii)
