import argparse
import pathlib
import os.path as osp
import sys

import open3d
import numpy as np

BASE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), osp.pardir)
sys.path.append(BASE_DIR)
sys.path.append(osp.join(BASE_DIR, 'src'))

from data.dynamic_point_cloud import FileBackedDynamicPointCloud
from downsample_ply import downsample_uniform, downsample

DEFAULT_POINTS = 2048
DEFAULT_METHOD = "uniform"
DEFAULT_ASCII = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help=f"Base directory of input.",
                        type=str)
    parser.add_argument("input_dpc",
                        help=f"Input dynamic pointcloud file.",
                        type=str)
    parser.add_argument("output_dir",
                        help=f"Base directory of output.",
                        type=str)
    parser.add_argument("output_dpc",
                        help=f"Output dynamic pointcloud file.",
                        type=str)
    parser.add_argument("output_name_scheme",
                        help=f"Name scheme used for output files [e.g. \"frame%02d.ply\"]",
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
    for arg in vars(args):
        k = arg
        v = getattr(args, arg)
        print(k, ": ", v)

    # Read in Dynamic Point Cloud
    input_dpc = FileBackedDynamicPointCloud(args.input_dir, args.input_dpc)

    # Create output dir if needed
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Keep track of all frame names
    frame_names = []

    # Downsample all the frames
    for (i, pc) in enumerate(input_dpc):
        pc_down = downsample(pc, args.method, args.points)
        output_frame_name = args.output_name_scheme % i
        output_frame_path = osp.join(args.output_dir, output_frame_name)
        frame_names.append(output_frame_name + "\n")

        # Write output PLY
        open3d.write_point_cloud(output_frame_path, pc_down, write_ascii=args.ascii)


    output_dpc_file = open(args.output_dpc, "w+")
    output_dpc_file.writelines(frame_names)
    output_dpc_file.close()

