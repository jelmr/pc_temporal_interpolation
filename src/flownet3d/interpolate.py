

import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import os.path as osp
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from dynamic_point_cloud import FileBackedDynamicPointCloud

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_concat_upsa', help='Model name [default: model_concat_upsa]')
parser.add_argument('--data', default='data_preprocessing/data_processed_maxcut_35_20k_2k_8192', help='Dataset directory [default: /data_preprocessing/data_processed_maxcut_35_20k_2k_8192]')
parser.add_argument('--model_path', default='log_train/model.ckpt', help='model checkpoint file path [default: log_train/model.ckpt]')
parser.add_argument('--log_dir', default='log_evaluate', help='Log dir [default: log_evaluate]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DATA = FLAGS.data
GPU_INDEX = FLAGS.gpu

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
MODEL_PATH = FLAGS.model_path
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(source, output, naming_scheme):
    # Input Dynamic Point Cloud

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, masks_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            print("--- Get model and loss")
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=None)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        ops = {'pointclouds_pl': pointclouds_pl,
               'masks_pl': masks_pl,
               'is_training_pl': is_training_pl,
               'pred': pred}

        interpolate(source, output, naming_scheme, sess, ops)


def normalize_np(pc):
    mean_ = np.mean(pc, axis=-2)
    mean_ = np.mean(mean_, axis=-2)
    mean_[3:] = 0
    pc -= mean_

    max_ = np.max(np.abs(pc))
    scale = (1./max_) * 0.99999
    pc[..., :3] *= scale

    return pc

def sample(x, n):
    pc1 = x[:, :(x.shape[1] // 2), :]
    pc2 = x[:, (x.shape[1] // 2):, :]

    batches, points, depth = pc1.shape

    pc_new = np.zeros((batches, 2*n, depth))

    for i in range(batches):
        idx1 = np.random.choice(points, n, replace=False)
        pc_new[i, :n, :] = pc1[i, idx1, :]
        idx2 = np.random.choice(points, n, replace=False)
        pc_new[i, n:, :] = pc2[i, idx2, :]

    return pc_new


def interpolate(source, output, naming_scheme, sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    # Test on all data: last batch might be smaller than BATCH_SIZE

    loss_sum = 0

    log_string(str(datetime.now()))
    log_string('---- EVALUATION ----')


    for (batch_idx, (pc1_batch, pc2_batch)) in enumerate(source.pair_batches(BATCH_SIZE)):
        print(f"Batch {batch_idx}")
        print(pc1_batch.shape)

        cur_batch_size = pc1_batch.shape[0]
        start_idx = batch_idx * BATCH_SIZE * 2

        num_points = pc1_batch.shape[1]

        batch_data = np.zeros((BATCH_SIZE, num_points*2, 6))
        batch_mask = np.ones((BATCH_SIZE, num_points))

        if cur_batch_size == BATCH_SIZE:
            batch_data[:, :num_points, :] = pc1_batch
            batch_data[:, num_points:, :] = pc2_batch
        else:
            batch_data[0:cur_batch_size, :num_points, :] = pc1_batch
            batch_data[0:cur_batch_size, num_points:, :] = pc2_batch

        batch_data = sample(batch_data, NUM_POINT)
        batch_data = normalize_np(batch_data)



        # ---------------------------------------------------------------------
        # ---- INFERENCE BELOW ----

        # TODO: TRY THIS
        # TODO: TRY THIS
        # TODO: TRY THIS
        # TODO: TRY THIS
        # TODO: TRY THIS
        # TODO: TRY THIS
        #pred_val_sum = np.zeros((BATCH_SIZE, NUM_POINT, 6))
        SHUFFLE_TIMES = 1
        for shuffle_cnt in range(SHUFFLE_TIMES):
            shuffle_idx = np.arange(NUM_POINT) # TODO: change this to indepently shuffle
            np.random.shuffle(shuffle_idx)
            batch_data_new = np.copy(batch_data)
            batch_data_new[:,0:NUM_POINT,:] = batch_data[:,shuffle_idx,:]
            batch_data_new[:,NUM_POINT:,:] = batch_data[:,NUM_POINT+shuffle_idx,:]
            feed_dict = {ops['pointclouds_pl']: batch_data_new,
                         ops['masks_pl']: batch_mask[:,shuffle_idx],
                         ops['is_training_pl']: is_training}
            pred_val = sess.run([ops['pred']], feed_dict=feed_dict)[0]
            #pred_val_sum[:,shuffle_idx,:] += tf.squeeze(pred_val)
        #pred_val_sum /= float(SHUFFLE_TIMES)
        # ---- INFERENCE ABOVE ----
        # ---------------------------------------------------------------------

        pc1_batch = batch_data[:, :NUM_POINT, :]
        pc2_batch = batch_data[:, NUM_POINT:, :]

        for (i, (pc1, pred, pc2)) in enumerate(zip(pc1_batch, pred_val, pc2_batch)):
            idx = start_idx + 2 * i
            output.add_np_pc(naming_scheme.format(idx), pc1)

            pc_new = pc1.copy()
            pc_new[..., :3] += pred[i, ...]
            output.add_np_pc(naming_scheme.format(idx + 1), pc_new)
            last_idx = idx + 2
            last_pc = pc2

    output.add_np_pc(naming_scheme.format(last_idx), last_pc)



if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))

    source_dpc = FileBackedDynamicPointCloud("/home/jelmer/data/Stefani_Capoeira", "/home/jelmer/data/Stefani_Capoeira/frames.dpc")

    # Output Dynamic Point Cloud
    base_dir = osp.join("/home/jelmer/data/interpolated/stefani_cap")
    output_dpc = FileBackedDynamicPointCloud(base_dir, osp.join(base_dir, "frames.dpc"))
    naming_scheme = osp.join(base_dir, "frame_{0:03d}.ply")

    # Perform the interpolation
    evaluate(source_dpc, output_dpc, naming_scheme)

    # Write results back to disk
    output_dpc.write_to_disk()
    LOG_FOUT.close()
