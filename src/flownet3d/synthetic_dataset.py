'''
    Provider for duck dataset from xingyu liu
'''

import os
import os.path
import json
import numpy as np
import sys
import pickle
import glob


class SceneflowDataset():
    def __init__(self, npoints=2048, train=True):
        self.npoints = npoints
        self.train = train
        if self.train:
            self.root = '/path/to/training_dataset_train.pickle'
        else:
            self.root = '/path/to/training_dataset_test.pickle'

        with open(self.root, 'rb') as fp:
            data = pickle.load(fp)
            self.pos1 = data['points1']
            self.pos2 = data['points2']
            self.pos3 = data['points3']
            self.color1 = data['color1'] #/ 255
            self.color2 = data['color2'] #/ 255
            self.color3 = data['color3'] #/ 255
            self.normals1 = data['normals1']
            self.mask1 = data['mask']


    def __getitem__(self, index):
        return (self.pos1[index, ...],
                self.pos2[index, ...],
                self.pos3[index, ...],
                self.color1[index, ...],
                self.color2[index, ...],
                self.color3[index, ...],
                self.normals1[index, ...],
                self.mask1[index, ...])


    def __len__(self):
        return self.pos1.shape[0]


if __name__ == '__main__':
    # import mayavi.mlab as mlab
    d = SceneflowDataset(npoints=2048)
    print(len(d))



