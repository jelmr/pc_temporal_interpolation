import open3d
import pathlib
from enum import Enum
import os.path as osp
import numpy as np
from utils.open3d_util import np_to_open3d_pc, open3d_to_np_pc


class FileBackedDynamicPointCloud:
    class Status(Enum):
        INVALID = 0
        CACHED = 1
        MODIFIED = 2

    class CacheLine:

        def __init__(self, file_name: str, pc):
            open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Warning)
            self.file_name_ = file_name
            self.pc_ = pc

            if pc:
                self.status_ = FileBackedDynamicPointCloud.Status.MODIFIED
            else:
                self.status_ = FileBackedDynamicPointCloud.Status.INVALID

    def __init__(self, data_dir: str, dpc_file_name: str, cache_size: int = 20, write_ascii: bool = True):
        self.frames_ = []
        self.lru_ = []
        self.cache_size_ = cache_size
        self.ascii_ = write_ascii
        self.dpc_file_name_ = dpc_file_name

        self.load_dynamic_point_cloud_(data_dir, dpc_file_name)

    def __getitem__(self, item):
        if item >= len(self.frames_):
            raise Exception(f'Dynamic point cloud has no frame {item}.')

        if self.frames_[item].status_ == FileBackedDynamicPointCloud.Status.INVALID:
            self.load_pc_(item)

        return self.frames_[item].pc_

    def __len__(self):
        return len(self.frames_)

    def __iter__(self):
        for i in range(len(self.frames_)):
            yield self[i]

    def pair_batches(self, batch_size, num_points=None):
        # Calculate how many batches we'll need
        num_triples = (len(self) - 1)
        full_batches = num_triples // batch_size

        # Determine dimensions
        points_np = np.asarray(self[0].points)
        colors_np = np.asarray(self[0].colors)

        if not num_points:
            num_points = points_np.shape[0]

        depth = points_np.shape[1] + colors_np.shape[1]  # TODO: always 6?

        num_frames_per_batch = batch_size + 1
        frames_np = np.zeros((num_frames_per_batch, num_points, depth))

        for batch_idx in range(full_batches):

            # Fill frames_np first to make it easier to generate triples batch
            # Avoids needing to worry about cache size.
            base_idx = batch_idx * (batch_size )
            for i in range(num_frames_per_batch):
                frame_idx = base_idx + i
                frames_np[i, :, :3] = np.asarray(self[frame_idx].points)[:num_points, :]
                frames_np[i, :, 3:] = np.asarray(self[frame_idx].colors)[:num_points, :]

            yield frames_np[:-1], np.copy(frames_np[1:])

    def triplet_batches(self, batch_size, num_points=None):

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
        # TODO: Also yield the remainder
        # TODO: Also yield the remainder
        # TODO: Also yield the remainder
        # TODO: Also yield the remainder

    def load_dynamic_point_cloud_(self, data_dir, dpc_file_name):
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
        self.frames_.append(FileBackedDynamicPointCloud.CacheLine(file_name, None))
        self.touch_cache(len(self)-1)

    def add_open3d_pc(self, file_name, pc):
        self.frames_.append(FileBackedDynamicPointCloud.CacheLine(file_name, pc))
        self.touch_cache(len(self)-1)

    def add_np_pc(self, file_name, pc):
        pc_open3d = np_to_open3d_pc(pc)
        self.add_open3d_pc(file_name, pc_open3d)

    def load_pc_(self, item):
        frame = self.frames_[item]
        frame.pc_ = open3d.read_point_cloud(frame.file_name_)
        frame.status_ = FileBackedDynamicPointCloud.Status.CACHED
        self.touch_cache(item)

    def touch_cache(self, item):

        # If it was already in cache queue, remove it
        if item in self.lru_:
            self.lru_.remove(item)

        # Add it at end
        self.lru_.append(item)

        # If cache is now too large, remove the oldest item
        if len(self.lru_) > self.cache_size_:
            self.unload_pc_(self.lru_[0])

    def set_modified(self, item):
        if item >= len(self.frames_):
            raise Exception(f'Dynamic point cloud has no frame {item}.')

        self.frames_[item].status_ = FileBackedDynamicPointCloud.Status.MODIFIED

    def write_to_disk(self):
        for frame_idx in range(len(self)):
            self.unload_pc_(frame_idx)
        self.update_dpc_file_()

    def update_dpc_file_(self):

        output_dpc_file = open(self.dpc_file_name_, "w+")

        output = ""
        for frame_idx in range(len(self)):
            frame = self.frames_[frame_idx]
            _, frame_file_name = osp.split(frame.file_name_)
            output += frame_file_name + "\n"

        output_dpc_file.write(output)
        output_dpc_file.close()

    def unload_pc_(self, item):
        frame = self.frames_[item]

        # If the PC was modified since being loaded from file
        if frame.status_ == FileBackedDynamicPointCloud.Status.MODIFIED:
            # Create output dir if needed
            output_dir, output_file = osp.split(frame.file_name_)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Write the PLY file
            open3d.write_point_cloud(frame.file_name_, frame.pc_, write_ascii=self.ascii_)

        # Delete it from the cache LRU queue if it's in there
        if item in self.lru_:
            self.lru_.remove(item)

        # Unload the PC itself
        frame.pc_ = None
        frame.status_ = FileBackedDynamicPointCloud.Status.INVALID
