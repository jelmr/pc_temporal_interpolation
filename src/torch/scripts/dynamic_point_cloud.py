#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Jelmer Mulder
# =============================================================================

import open3d
from collections import defaultdict
import pathlib
from enum import Enum
import os.path as osp
import numpy as np
import sys

BASE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), osp.pardir)
sys.path.append(BASE_DIR)
sys.path.append(osp.join(BASE_DIR, 'src'))
sys.path.append(osp.join(BASE_DIR, 'src', 'data'))
sys.path.append(osp.join(BASE_DIR, 'src', 'utils'))

#from utils.open3d_util import np_to_open3d_pointcloud as np_to_open3d_pc
#from utils.open3d_util import open3d_to_np_pointcloud as open3d_to_np_pc
from utils.open3d_util import np_to_open3d_pc, open3d_to_np_pc


class FileBackedDynamicPointCloud:
    class Status(Enum):
        INVALID = 0
        CACHED = 1
        MODIFIED = 2

    class CacheLine:

        def __init__(self, file_name: str, pc):
            self._file_name = file_name
            self._pc = pc

            if pc:
                self._status = FileBackedDynamicPointCloud.Status.MODIFIED
            else:
                self._status = FileBackedDynamicPointCloud.Status.INVALID

    def __init__(self, data_dir: str, dpc_file_name: str, cache_size: int = 20, write_ascii: bool = True):
        self._frames = []
        self._lru = []
        self._cache_size = cache_size
        self._ascii = write_ascii
        self._dpc_file_name = dpc_file_name

        self._load_dynamic_point_cloud(data_dir, dpc_file_name)

    def __getitem__(self, item):
        if item >= len(self._frames):
            raise Exception(f'Dynamic point cloud has no frame {item}.')

        if self._frames[item]._status == FileBackedDynamicPointCloud.Status.INVALID:
            self._load_pc(item)

        return self._frames[item]._pc

    def __len__(self):
        return len(self._frames)

    def __iter__(self):
        for i in range(len(self._frames)):
            yield self[i]

    def triplet_batches(self, batch_size, num_points=None):
        # TODO: This function does not belong in this class and
        #   should be moved elsewhere.

        # Calculate how many batches we'll need
        num_triples = (len(self) - 1) // 2
        full_batches = num_triples // batch_size

        # Determine dimensions
        points_np = np.asarray(self[0].points)
        colors_np = np.asarray(self[0].colors)

        if not num_points:
            num_points = points_np.shape[0]

        depth = points_np.shape[1] + colors_np.shape[1]  # TODO: always 6?

        num_frames_per_batch = 2 * batch_size + 1
        frames_np = np.zeros((num_frames_per_batch, num_points, depth))

        for batch_idx in range(full_batches):

            # Fill frames_np first to make it easier to generate triples batch
            # Avoids needing to worry about cache size.
            base_idx = batch_idx * batch_size * 2
            for i in range(num_frames_per_batch):
                frame_idx = base_idx + i
                frames_np[i, :, :3] = np.asarray(self[frame_idx].points)[:num_points, :]
                frames_np[i, :, 3:] = np.asarray(self[frame_idx].colors)[:num_points, :]

            yield frames_np[:-2:2], np.copy(frames_np[1:-1:2]), np.copy(frames_np[2::2])

        # TODO: Also yield the remainder

    def _load_dynamic_point_cloud(self, data_dir, dpc_file_name):
        dpc_file_path = dpc_file_name  # osp.join(data_dir, dpc_file_name) # TODO: Clean
        if osp.exists(dpc_file_path):
            # If the DPC file exist, open it and load all referenced PLYs
            dpc_file = open(dpc_file_path, "r")
            pc_files = dpc_file.read().splitlines()

            for pc_file in pc_files:
                pc_path = osp.join(data_dir, pc_file)
                self.add_file(pc_path)

            dpc_file.close()
        else: # If file does not exist
            # Create output dir if needed
            output_dir, output_file = osp.split(dpc_file_path)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Create DPC file itself
            dpc_file = open(dpc_file_path, "w")
            dpc_file.close()

    def add_file(self, file_name):
        self._frames.append(FileBackedDynamicPointCloud.CacheLine(file_name, None))
        self.touch_cache(len(self)-1)

    def add_open3d_pc(self, file_name, pc):
        self._frames.append(FileBackedDynamicPointCloud.CacheLine(file_name, pc))
        self.touch_cache(len(self)-1)

    def add_np_pc(self, file_name, pc):
        pc_open3d = np_to_open3d_pointcloud(pc)
        self.add_open3d_pc(file_name, pc_open3d)

    def _load_pc(self, item):
        frame = self._frames[item]
        frame._pc = open3d.read_point_cloud(frame._file_name)
        frame._status = FileBackedDynamicPointCloud.Status.CACHED
        self.touch_cache(item)

    def touch_cache(self, item):

        # If it was already in cache queue, remove it
        if item in self._lru:
            self._lru.remove(item)

        # Add it at end
        self._lru.append(item)

        # If cache is now too large, remove the oldest item
        if len(self._lru) > self._cache_size:
            self._unload_pc(self._lru[0])

    def set_modified(self, item):
        if item >= len(self._frames):
            raise Exception(f'Dynamic point cloud has no frame {item}.')

        self._frames[item]._status = FileBackedDynamicPointCloud.Status.MODIFIED

    def write_to_disk(self):
        for frame_idx in range(len(self)):
            self._unload_pc(frame_idx)
        self._update_dpc_file()

    def _update_dpc_file(self):

        output_dpc_file = open(self._dpc_file_name, "w+")

        output = ""
        for frame_idx in range(len(self)):
            frame = self._frames[frame_idx]
            _, frame_file_name = osp.split(frame.file_name_)
            output += frame_file_name + "\n"

        output_dpc_file.write(output)
        output_dpc_file.close()

    def _unload_pc(self, item):
        frame = self._frames[item]

        # If the PC was modified since being loaded from file
        if frame._status == FileBackedDynamicPointCloud.Status.MODIFIED:
            # Create output dir if needed
            output_dir, output_file = osp.split(frame.file_name_)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Write the PLY file
            open3d.write_point_cloud(frame.file_name_, frame._pc, write_ascii=self._ascii)

        # Delete it from the cache LRU queue if it's in there
        if item in self._lru:
            self._lru.remove(item)

        # Unload the PC itself
        frame._pc = None
        frame._status = FileBackedDynamicPointCloud.Status.INVALID
