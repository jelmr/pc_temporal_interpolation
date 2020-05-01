#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Jelmer Mulder
# =============================================================================

import open3d
import time
from timeit import default_timer as timer
import sys
import pathlib
import numpy as np
import copy
import os
import argparse
import os.path as osp
from dynamic_point_cloud import FileBackedDynamicPointCloud

FRAME_NAME_PATTERN = 'frame%04d.jpg'
DEFAULT_FPS = 10
DEFAULT_LOOP = False
DEFAULT_BLACK = False
DEFAULT_SBS = False
DEFAULT_VIDEO = False


class DynamicPointCloudRenderer:

    def __init__(self,
                 fps: int = DEFAULT_FPS,
                 loop: bool = False):
        self._fps = fps
        self.frame_duration = (1. / fps)
        self._vis = None
        self._active_pc = None
        self._loop = loop

    def render(self,
               dpc: FileBackedDynamicPointCloud,
               img_output_dir: str = None,
               color_black: bool = False,
               interpolate: int = 0,
               sbs: bool = False):

        if not self._vis:
            self._vis = open3d.Visualizer()
            self._vis.create_window()
            # self._active_pc = open3d.geometry.PointCloud()
            self._active_pc = dpc[0]
            self._vis.add_geometry(self._active_pc)

            if sbs:
                self._sbs_pc = open3d.geometry.PointCloud()
                self._sbs_pc.points = open3d.Vector3dVector(np.asarray(self._active_pc.points).copy())
                self._sbs_pc.colors = open3d.Vector3dVector(np.asarray(self._active_pc.colors).copy())

                active_x_min = np.min(np.asarray(self._active_pc.points)[..., 0])
                sbs_x_max = np.max(np.asarray(self._sbs_pc.points)[..., 0])
                self._sbs_offset = np.asarray([2 * (sbs_x_max - active_x_min),0,0])
                np.asarray(self._sbs_pc.points)[...] += self._sbs_offset

                self._vis.add_geometry(self._sbs_pc)



        if img_output_dir:
            pathlib.Path(img_output_dir).mkdir(parents=True, exist_ok=True)

        while True:
            for (i, frame) in enumerate(dpc):
                print(f"Frame {i}")
                normals = copy.deepcopy(frame.normals)
                frame.normals = open3d.Vector3dVector([])


                for j in range(interpolate+1):

                    if j == 0 and sbs:
                        self._sbs_pc.points = open3d.Vector3dVector(np.asarray(self._active_pc.points + self._sbs_offset).copy())
                        self._sbs_pc.colors = open3d.Vector3dVector(np.asarray(self._active_pc.colors).copy())

                    if j > 0:
                        x = np.asarray(normals) / (interpolate+1)
                        np.asarray(frame.points)[...] += x

                    time_start = timer()

                    if color_black:
                        cols = np.asarray(frame.colors)
                        cols[...] = 0

                    # Draw the new point cloud
                    self.set_active_frame(frame)
                    self._vis.update_geometry()
                    self._vis.update_renderer()
                    self._vis.poll_events()
                    self._vis.run()

                    if img_output_dir:
                        frame_idx = (interpolate+1) * i + j
                        img_file_name = osp.join(img_output_dir, FRAME_NAME_PATTERN % frame_idx)
                        self._vis.capture_screen_image(img_file_name)

                    # Sleep as long as needed to ensure the desired FPS
                    time_end = timer()
                    time_diff = time_end - time_start
                    time_remaining = (self.frame_duration / (interpolate+1)) - time_diff
                    if time_remaining > 0:
                        time.sleep(time_remaining)
                    elif i >= 1:  # First frame tends to be slow, don't output warning for it.
                        current_fps = 1 / time_diff
                        print(f"Warning: Not running at full FPS. (Current FPS: {current_fps:.2f})")
            if not self._loop:
                self._vis.destroy_window()
                break

    def set_active_frame(self, frame):
        #frame = open3d.voxel_down_sample(frame, voxel_size=0.02)
        self._active_pc.points = frame.points
        self._active_pc.colors = frame.colors
        a = np.asarray(self._active_pc.colors)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help=f"Base directory of input.",
                        type=str)
    parser.add_argument("input_dpc",
                        help=f"Input dynamic pointcloud file.",
                        type=str)
    parser.add_argument("--img_output_dir",
                        help=f"If provided, images of the rendering will be stored in this directory.",
                        type=str,
                        default=None)
    parser.add_argument("--fps",
                        help=f"Frames per second to render the frames in [default={DEFAULT_FPS}]",
                        type=float,
                        default=DEFAULT_FPS)
    parser.add_argument("--interpolate",
                        help=f"Number of frames to interpolate, only possible if normals are present.",
                        type=int,
                        default=0)
    parser.add_argument("--loop",
                        help=f"If true, starts from beginning once finished [default={DEFAULT_LOOP}",
                        action="store_true",
                        default=DEFAULT_LOOP)
    parser.add_argument("--black",
                        help=f"Render the entire point cloud black. [default={DEFAULT_BLACK}",
                        action="store_true",
                        default=DEFAULT_BLACK)
    parser.add_argument("--sbs",
                        help=f"Render the raw and interpolated versions side-by-side. [default={DEFAULT_SBS}",
                        action="store_true",
                        default=DEFAULT_SBS)
    parser.add_argument("--video",
                        help=f"Create a video from the images. [default={DEFAULT_VIDEO}",
                        action="store_true",
                        default=DEFAULT_VIDEO)
    return parser.parse_args()


if __name__ == "__main__":
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Warning)
    args = parse_args()

    if args.loop and args.img_output_dir:
        print("Cannot use '--loop' and '--img_output_dir' at the same time.")
        sys.exit(1)

    dpc = FileBackedDynamicPointCloud(args.input_dir, args.input_dpc)

    rd = DynamicPointCloudRenderer(fps=args.fps, loop=args.loop)

    rd.render(dpc, img_output_dir=args.img_output_dir, color_black=args.black, interpolate=args.interpolate, sbs=args.sbs)

    if args.video:
        os.chdir(args.img_output_dir)
        os.system(f"ffmpeg -r {args.fps * (args.interpolate+1)} -i \"{FRAME_NAME_PATTERN}\" -c:v libx264 -crf 15 vid.mp4")
