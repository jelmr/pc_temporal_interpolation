'''

'''

import os
import os.path
import os.path as osp
import json
import numpy as np
import sys
import pickle
import glob

sys.path.append("/home/jelmer/pc_interpolation")
import basic_data_provider as provider


class SceneflowDataset():
    def __init__(self,
            input_dir = "/home/jelmer/pc_interpolation/data/modelnet40_ply_hdf5_2048",
            npoints=2048,
            train=True):
        self.npoints = npoints
        self.train = train


        self.input_file = osp.join(input_dir, "ply_data_train0.h5")
        self.names_file = osp.join(input_dir, "shape_names.txt")

        self.reload()


    def reload(self):
        input_file = self.input_file
        names_file = self.names_file
        objs = provider.get_object_from_input(input_file, names_file)
        pc1, label, pc2 = provider.get_video_style_pc_permutations(objs)
        self.pos1 = pc1[..., :3]
        self.pos2 = label[..., :3]
        self.pos3 = pc2[..., :3]
        self.color1 = pc1[..., 3:6]
        self.color2 = label[..., 3:6]
        self.color3 = pc2[..., 3:6]
        self.normals1 = self.pos2 - self.pos1




    def __getitem__(self, index):
        return (self.pos1[index, ...],
                self.pos2[index, ...],
                self.pos2[index, ...],
                self.color1[index, ...],
                self.color2[index, ...],
                self.color2[index, ...],
                self.normals1[index, ...],
                np.zeros((self.pos1.shape[1]))
                )


    def __len__(self):
        return self.pos1.shape[0]


if __name__ == '__main__':
    # import mayavi.mlab as mlab
    d = SceneflowDataset(npoints=2048)



