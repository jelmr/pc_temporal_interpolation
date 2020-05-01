import pickle
import numpy as np
import copy

INPUT_DATASET = "/path/to/torch/data/SyntheticAdvanced/processed/training_data.pickle"
OUTPUT_TRAIN = "/some/path/training_data_train.pickle"
OUTPUT_TEST = "/some/path/training_data_test.pickle"
NUM_TEST = 500 # Number of samples used for the test set, the rest is used for training

f = open(INPUT_DATASET, "rb")
d = pickle.load(f)
TRAIN = {}
TEST = {}

keys = list(d.keys())

s = d["points1"].shape
l = s[0]
shuffle_idx = np.arange(l)
np.random.shuffle(shuffle_idx)

for key in keys:
    shape = list(d[key].shape)
    shape[0]= shape[0] - NUM_TEST
    new_train = np.zeros(shape)
    new_train[...] = d[key][shuffle_idx[:-NUM_TEST:1], ...]
    TRAIN[key] = new_train

    shape[0]= NUM_TEST
    new_test = np.zeros(shape)
    new_test[...] = d[key][shuffle_idx[-NUM_TEST::1], ...]
    TEST[key] = new_test



f = open(OUTPUT_TRAIN, "wb")
pickle.dump(TRAIN, f)
f.close()

f = open(OUTPUT_TEST, "wb")
pickle.dump(TEST, f)
f.close()

