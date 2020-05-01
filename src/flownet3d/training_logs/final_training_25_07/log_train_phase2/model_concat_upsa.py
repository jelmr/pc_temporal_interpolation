"""
    FlowNet3D model with up convolution
"""

import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
from pointnet_util import *

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * 2, 6))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    masks_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, masks_pl
def get_old_model(point_cloud, is_training, bn_decay=None):

    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // 2

    l0_xyz_f1 = point_cloud[:, :num_point, 0:3]
    l0_points_f1 = point_cloud[:, :num_point, 3:]
    l0_xyz_f2 = point_cloud[:, num_point:, 0:3]
    l0_points_f2 = point_cloud[:, num_point:, 3:]

    CM = 1/100
    # RADIUS1 = 0.5
    # RADIUS2 = 1.0
    # RADIUS3 = 2.0
    # RADIUS4 = 4.0
    RADIUS1 = 8 * CM
    RADIUS2 = 15 * CM
    RADIUS3 = 25 * CM
    RADIUS4 = 35 * CM
    RADIUS_FLOW = 35*CM
    with tf.variable_scope('sa1') as scope:
        # Frame 1, Layer 1
        l1_xyz_f1, l1_points_f1, l1_indices_f1 = pointnet_sa_module(l0_xyz_f1, l0_points_f1, npoint=32, radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_points=False)

        end_points['l1_indices_f1'] = l1_indices_f1
        end_points['l1_xyz_f1'] = l1_xyz_f1
        end_points['l1_points_f1'] = l1_points_f1

        # Frame 1, Layer 2
        # l2_xyz_f1, l2_points_f1, l2_indices_f1 = pointnet_sa_module(l1_xyz_f1, l1_points_f1, npoint=256, radius=RADIUS2, nsample=16, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        # end_points['l2_indices_f1'] = l2_indices_f1

        scope.reuse_variables()
        # Frame 2, Layer 1
        l1_xyz_f2, l1_points_f2, l1_indices_f2 = pointnet_sa_module(l0_xyz_f2, l0_points_f2, npoint=32  , radius=RADIUS1, nsample=16, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_points=False)
        # # Frame 2, Layer 2
        # l2_xyz_f2, l2_points_f2, l2_indices_f2 = pointnet_sa_module(l1_xyz_f2, l1_points_f2, npoint=256, radius=RADIUS2, nsample=16, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    _, l2_points_f1_new = flow_embedding_module(l1_xyz_f1, l1_xyz_f2, l1_points_f1, l1_points_f2, radius=RADIUS_FLOW, nsample=32, mlp=[128,128,128], is_training=is_training, bn_decay=bn_decay, scope='flow_embedding', bn=True, pooling='max', knn=True, corr_func='concat')

    # Layer 3
    l3_xyz_f1, l3_points_f1, l3_indices_f1 = pointnet_sa_module(l1_xyz_f1, l2_points_f1_new, npoint=64, radius=RADIUS3, nsample=8, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    end_points['l3_indices_f1'] = l3_indices_f1

    # Layer 4
    l4_xyz_f1, l4_points_f1, l4_indices_f1 = pointnet_sa_module(l3_xyz_f1, l3_points_f1, npoint=16, radius=RADIUS4, nsample=8, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    end_points['l4_indices_f1'] = l4_indices_f1

    # Feature Propagation
    l3_feat_f1 = set_upconv_module(l3_xyz_f1, l4_xyz_f1, l3_points_f1, l4_points_f1, nsample=8, radius=2.4, mlp=[], mlp2=[256,256], scope='up_sa_layer1', is_training=is_training, bn_decay=bn_decay, knn=True)
    l2_feat_f1 = set_upconv_module(l1_xyz_f1, l3_xyz_f1, tf.concat(axis=-1, values=[l1_points_f1, l2_points_f1_new]), l3_feat_f1, nsample=8, radius=1.2, mlp=[128,128,256], mlp2=[256], scope='up_sa_layer2', is_training=is_training, bn_decay=bn_decay, knn=True)
    # l1_feat_f1 = set_upconv_module(l1_xyz_f1, l1_xyz_f1, l1_points_f1, l2_feat_f1, nsample=8, radius=0.6, mlp=[128,128,256], mlp2=[256], scope='up_sa_layer3', is_training=is_training, bn_decay=bn_decay, knn=True)
    l0_feat_f1 = pointnet_fp_module(l0_xyz_f1, l1_xyz_f1, l0_points_f1, l2_feat_f1, [256,256], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_feat_f1, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc2')


    #pc1 = point_cloud[:, :num_point, :]
    #return pc1+net, end_points
    return net, end_points

    #return net, end_points






def get_model(point_cloud, is_training, bn_decay=None):

    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // 2

    pc1_xyz = point_cloud[:, :num_point, 0:3]
    pc1_points = point_cloud[:, :num_point, 3:]
    pc2_xyz = point_cloud[:, num_point:, 0:3]
    pc2_points = point_cloud[:, num_point:, 3:]

    CM = 1/100
    RADIUS1 = 8 * CM
    RADIUS2 = 15 * CM
    RADIUS3 = 25 * CM
    RADIUS4 = 35 * CM
    RADIUS_FLOW = 15*CM
    with tf.variable_scope('sa1') as scope:
        ############
        # DOWNSAMPLING PC1
        ##################
        print("Defining DOWNSAMPLING PC1")


        # pc1_512_xyz, pc1_512_points, pc1_512_idx = pointnet_sa_module(pc1_xyz, pc1_points,
        #                                                             npoint=512, radius=RADIUS1, nsample=24,
        #                                                             mlp=None, mlp2=None,
        #                                                             group_all=False, is_training=is_training,
        #                                                             bn_decay=bn_decay, scope='layer1', use_points=False)
        # pc1_128_xyz, pc1_128_points, pc1_128_idx = pointnet_sa_module(pc1_xyz, pc1_points,
        #                                                                  npoint=128, radius=RADIUS2, nsample=6,
        #                                                                  mlp=None, mlp2=None,
        #                                                                  group_all=False, is_training=is_training,
        #                                                                  bn_decay=bn_decay, scope='layer2', use_points=False)
        # pc1_32_xyz, pc1_32_points, pc1_32_idx = pointnet_sa_module(pc1_xyz, pc1_points,
        #                                                                  npoint=32, radius=RADIUS3, nsample=3,
        #                                                                  mlp=None, mlp2=None,
        #                                                                  group_all=False, is_training=is_training,
        #                                                                  bn_decay=bn_decay, scope='layer3', use_points=False)


        scope.reuse_variables()
        ##################
        # DOWNSAMPLING PC2
        ##################
        print("Defining DOWNSAMPLING PC2")
        # pc2_512_xyz, pc2_512_points, pc2_512_idx = pointnet_sa_module(pc2_xyz, pc2_points,
        #                                                             npoint=512, radius=RADIUS1, nsample=24,
        #                                                             mlp=None, mlp2=None,
        #                                                             group_all=False, is_training=is_training,
        #                                                             bn_decay=bn_decay, scope='layer1', use_points=False)
        # pc2_128_xyz, pc2_128_points, pc2_128_idx = pointnet_sa_module(pc2_xyz, pc2_points,
        #                                                               npoint=128, radius=RADIUS2, nsample=6,
        #                                                               mlp=None, mlp2=None,
        #                                                               group_all=False, is_training=is_training,
        #                                                               bn_decay=bn_decay, scope='layer2', use_points=False)
        # pc2_32_xyz, pc2_32_points, pc2_32_idx = pointnet_sa_module(pc2_xyz, pc2_points,
        #                                                               npoint=32, radius=RADIUS3, nsample=3,
        #                                                               mlp=None, mlp2=None,
        #                                                               group_all=False, is_training=is_training,
        #                                                               bn_decay=bn_decay, scope='layer3', use_points=False)


        ##################
        # FLOW EMBEDDINGS
        ##################
        print("Defining FLOW EMBEDDINGS")
    _, flow_2048, sm_2048, neighbours_2048, o, feat_diff, xyz_diff = softmax_embedding(pc1_xyz, pc2_xyz, pc1_points, pc2_points,
                                     radius=RADIUS_FLOW, nsample=28,
                                         mlp=[128,128,64,1],
                                     is_training=is_training, bn_decay=bn_decay, scope='flow_embedding2048',
                                     bn=True, knn=True, corr_func='concat')
    # _, flow_512, sm_512, neighbours_512 = softmax_embedding(pc1_512_xyz, pc2_512_xyz, pc1_512_points, pc2_512_points,
    #                                 radius=RADIUS_FLOW, nsample=20,
    #                                 mlp=[128,64,64,1],
    #                                 is_training=is_training, bn_decay=bn_decay, scope='flow_embedding512',
    #                                 bn=True, knn=True, corr_func='concat')
    # _, flow_128, sm_128, neighbours_128 = softmax_embedding(pc1_128_xyz, pc2_128_xyz, pc1_128_points, pc2_128_points,
    #                                 radius=RADIUS_FLOW, nsample=12,
    #                                 mlp=[128,64,32,1],
    #                                 is_training=is_training, bn_decay=bn_decay, scope='flow_embedding128',
    #                                 bn=True, knn=True, corr_func='concat')
    # _, flow_32, sm_32, neighbours_32 = softmax_embedding(pc1_32_xyz, pc2_32_xyz, pc1_32_points, pc2_32_points,
    #                                 radius=RADIUS_FLOW, nsample=4,
    #                                 mlp=[128,64,32,1],
    #                                 is_training=is_training, bn_decay=bn_decay, scope='flow_embedding32',
    #                                 bn=True, knn=True, corr_func='concat')


    ##################
    # UPSAMPLING
    ##################
    print("Defining UPSAMPLING")
    # == OLD UPSAMPLING ==
    # flow_32_128 = set_upconv_module(pc1_128_xyz, pc1_32_xyz, flow_128, flow_32,
    #                                nsample=8, radius=RADIUS4,
    #                                mlp=[], mlp2=[256,256, 3],
    #                                scope='up_sa_layer1', is_training=is_training,
    #                                bn_decay=bn_decay, knn=True)
    # flow_128_512 = set_upconv_module(pc1_512_xyz, pc1_128_xyz, flow_512, flow_32_128,
    #                                 nsample=16, radius=RADIUS3,
    #                                 mlp=[], mlp2=[256, 256, 3],
    #                                 scope='up_sa_layer2', is_training=is_training,
    #                                 bn_decay=bn_decay, knn=True)
    # flow_512_2048 = set_upconv_module(pc1_xyz, pc1_512_xyz, flow_2048, flow_128_512,
    #                                  nsample=32, radius=RADIUS2,
    #                                  mlp=[], mlp2=[256, 256, 3],
    #                                  scope='up_sa_layer3', is_training=is_training,
    #                                  bn_decay=bn_decay, knn=True)

    # flow_32_128, smu_128, idx128, xyz_diff_128 = upconv_fixed(pc1_128_xyz, pc1_32_xyz, flow_128, flow_32,
    #                                nsample=8, radius=RADIUS4,
    #                                mlp=[128, 64, 64, 1], mlp2=[],
    #                                scope='up_sa_layer1', is_training=is_training,
    #                                bn_decay=bn_decay, knn=True)
    # flow_128_512, smu_512, idx512, xyz_diff_512 = upconv_fixed(pc1_512_xyz, pc1_128_xyz, flow_512, flow_32_128,
    #                                 nsample=16, radius=RADIUS3,
    #                                 mlp=[128, 128, 64, 1], mlp2=[],
    #                                 scope='up_sa_layer2', is_training=is_training,
    #                                 bn_decay=bn_decay, knn=True)
    # flow_512_2048, smu_2048, idx2048, xyz_diff_2048 = upconv_fixed(pc1_xyz, pc1_512_xyz, flow_2048, flow_128_512,
    #                                  nsample=32, radius=RADIUS2,
    #                                  mlp=[128, 128, 64, 1], mlp2=[],
    #                                  scope='up_sa_layer3', is_training=is_training,
    #                                  bn_decay=bn_decay, knn=True)

    # end_points["flow_2048"] = flow_2048 # B x N x 3
    # end_points["flow_512"] = flow_512
    # end_points["flow_128"] = flow_128
    # end_points["flow_32"] = flow_32
    # end_points["neighbours_2048"] = neighbours_2048 # B x N x K x 3
    # end_points["neighbours_512"] = neighbours_512
    # end_points["neighbours_128"] = neighbours_128
    # end_points["neighbours_32"] = neighbours_32
    # end_points["sm_2048"] = sm_2048 # B x N x K
    # end_points["sm_512"] = sm_512
    # end_points["sm_128"] = sm_128
    # end_points["sm_32"] = sm_32
    # end_points["smu_2048"] = smu_2048 # B x N x K
    # end_points["smu_512"] = smu_512
    # end_points["smu_128"] = smu_128
    # end_points["idx_2048"] = idx2048 # B x N x K
    # end_points["idx_512"] = idx512
    # end_points["idx_128"] = idx128
    # end_points["xyz_2048_2"] = pc2_xyz
    # end_points["xyz_512_2"] = pc2_512_xyz
    # end_points["xyz_128_2"] = pc2_128_xyz
    # end_points["xyz_32_2"] = pc2_32_xyz
    # end_points["xyz_2048_1"] = pc1_xyz
    # end_points["xyz_512_1"] = pc1_512_xyz
    # end_points["xyz_128_1"] = pc1_128_xyz
    # end_points["xyz_32_1"] = pc1_32_xyz
    #
    # end_points["flow_2048_out"] = flow_512_2048
    # end_points["flow_512_out"] = flow_128_512
    # end_points["flow_128_out"] = flow_32_128
    # end_points["flow_32_out"] = flow_32
    #
    # end_points["xyz_diff_2048"] = xyz_diff_2048
    # end_points["xyz_diff_512"] = xyz_diff_512
    # end_points["xyz_diff_128"] = xyz_diff_128

    ##################
    # FULLY CONNECTED
    ##################

    # flow_out = flow_filter_module(pc1_xyz, flow_2048,
    #                        radius=RADIUS_FLOW, nsample=24,
    #                        mlp=[128,64,64],
    #                        is_training=is_training, bn_decay=bn_decay, scope='flow_filter',
    #                        bn=True, knn=True, corr_func='concat')


    # end_points["flow_2048"] = flow_2048
    # end_points["sm_2048"] = sm_2048
    # end_points["n_2048"] = neighbours_2048
    # end_points["o"] = o
    # end_points["xyz_diff"] = xyz_diff
    # end_points["feat_diff"] = feat_diff
    # end_points["flow_out"] = flow_out
    # end_points["flow_out"] = flow_out

    print("Defining FULLY CONNECTED")
    # net = tf_util.conv1d(flow_out, 64, 1, padding='VALID',
    #                      bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    #
    # end_points["fc1"] = net

    #net = tf.concat(axis=-1, values=[net, flow_2048]) # batch_size, npoint, nsample, channel*2

    # net = tf_util.conv1d(net, 3, 1, padding='VALID',
    #                      activation_fn=None, scope='fc2')
    # end_points["fc2"] = net


    print("GRAPH DEFINED")
    # return net, end_points
    return flow_2048, end_points
    # return flow_512_2048, end_points

def get_flow_refine_model(point_cloud, flow_2048, is_training, bn_decay=None):

    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // 2

    pc1_xyz = point_cloud[:, :num_point, 0:3]
    pc1_points = point_cloud[:, :num_point, 3:]
    pc2_xyz = point_cloud[:, num_point:, 0:3]
    pc2_points = point_cloud[:, num_point:, 3:]

    CM = 1/100
    RADIUS1 = 8 * CM
    RADIUS2 = 15 * CM
    RADIUS3 = 25 * CM
    RADIUS4 = 35 * CM
    RADIUS_FLOW = 15*CM

    # _, flow_2048, sm_2048, neighbours_2048, o, feat_diff, xyz_diff = softmax_embedding(pc1_xyz, pc2_xyz, pc1_points, pc2_points,
    #                                                                                    radius=RADIUS_FLOW, nsample=28,
    #                                                                                    mlp=[128,64,64,1],
    #                                                                                    is_training=is_training, bn_decay=bn_decay, scope='flow_embedding2048',
    #                                                                                    bn=True, knn=True, corr_func='concat')

    with tf.variable_scope('flow_refine') as scope:
        flow_out = flow_filter_module(pc1_xyz, flow_2048,
                                      radius=RADIUS_FLOW, nsample=12,
                                      mlp=[128,64,64], mlp2=[128,64,64,1],
                                      is_training=is_training, bn_decay=bn_decay, scope='flow_filter',
                                      bn=True, knn=True, corr_func='concat')


    # end_points["flow_2048"] = flow_2048
    # end_points["sm_2048"] = sm_2048
    # end_points["n_2048"] = neighbours_2048
    # end_points["o"] = o
    # end_points["xyz_diff"] = xyz_diff
    # end_points["feat_diff"] = feat_diff
    # end_points["flow_out"] = flow_out
    # end_points["flow_out"] = flow_out

        print("Defining FULLY CONNECTED")
        # fc1 = tf_util.conv1d(flow_out, 64, 1, padding='VALID',
        #                      bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)

        # end_points["fc1"] = fc1
        #
        # fc1 = tf.concat(axis=-1, values=[fc1, flow_2048]) # batch_size, npoint, nsample, channel*2
        #
        # fc2 = tf_util.conv1d(fc1, 3, 1, padding='VALID',
        #                      activation_fn=None, scope='fc2')
        # end_points["fc2"] = fc2


        print("GRAPH DEFINED")
    # return net, end_points
    # return flow_2048 + fc2, end_points
    return flow_out, end_points
    # return flow_512_2048, end_points





def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses)


def get_loss(pred, label, mask, end_points):
    """ pred: BxNx3,
        label: BxNx3,
        mask: BxN
    """
    batch_size = pred.get_shape()[0].value
    num_point = pred.get_shape()[1].value
    l2_loss = tf.reduce_mean(tf.reduce_sum((pred-label) * (pred-label), axis=2) / 2.0)
    tf.summary.scalar('l2 loss', l2_loss)
    tf.add_to_collection('losses', l2_loss)
    return l2_loss


# def get_loss(pred, label, normals, mask, end_points):
#     loss1 = get_oneway_loss(pred, label)
#     loss2 = get_oneway_loss(label, pred)
#     p2point_loss =  tf.reduce_mean(loss1 + loss2)
#     p2plane_loss =  tf.reduce_mean(get_p2plane_loss(pred, label, normals))
#     return (p2plane_loss**2 + (p2point_loss/5)**2) ** 0.5
#     # return tf.reduce_mean(get_p2plane_loss(pred, label, normals))


def get_oneway_loss(pc1, pc2):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # Ensure inputs are the same shape

    num_points_1 = pc1.get_shape()[1].value
    num_points_2 = pc2.get_shape()[1].value

    batch_size = pc1.get_shape()[0].value

    pc1 = tf.expand_dims(pc1, 1)
    pc2 = tf.expand_dims(pc2, 1)

    # Calculate the adjacency matrix between 'a' and 'b'. Here
    # we only use the first 3 coordinates, as these represent x,y,z.
    loc_pc1 = pc1[..., :3]
    loc_pc2 = pc2[..., :3]

    pc1_tiled = tf.tile(loc_pc1, (1, num_points_2, 1, 1))
    pc1_tiled = tf.reshape(pc1_tiled, [batch_size, num_points_2, num_points_1, 3])

    pc2_tiled = tf.tile(loc_pc2, (1, num_points_1, 1, 1))
    pc2_tiled = tf.reshape(pc2_tiled, [batch_size, num_points_1, num_points_2, 3])

    pc1_tiled_t = tf.transpose(pc1_tiled, perm=[0, 2, 1, 3])

    dist = pc1_tiled_t - pc2_tiled
    dist_squared = tf.square(dist)
    dist_squared_sum = tf.reduce_sum(dist_squared, axis=-1, keepdims=True)
    adj = tf.sqrt(dist_squared_sum)

    # For each poin in 'pc1' get the closest point in 'pc2'
    neg_adj = -adj

    min_idx = tf.argmax(neg_adj, axis=-2)

    idx_ = tf.range(batch_size, dtype=tf.int64) * num_points_2
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])
    pc2_flat = tf.reshape(pc2, [-1, 6])

    nearest_neighbours = tf.gather(pc2_flat, min_idx + idx_, axis=0)

    # For each pair of points, calculate the squared error.
    point_distance = tf.squeeze(pc1) - tf.squeeze(nearest_neighbours)
    point_error = tf.reduce_sum(tf.square(point_distance), axis=-1, keepdims=True)

    means = tf.reduce_mean(point_error, axis=1)
    return means

def get_p2plane_loss(pc1, pc2, normal):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # Ensure inputs are the same shape

    num_points_1 = pc1.get_shape()[1].value
    num_points_2 = pc2.get_shape()[1].value

    batch_size = pc1.get_shape()[0].value

    pc1 = tf.expand_dims(pc1, 1)
    pc2 = tf.expand_dims(pc2, 1)

    # Calculate the adjacency matrix between 'a' and 'b'. Here
    # we only use the first 3 coordinates, as these represent x,y,z.
    loc_pc1 = pc1[..., :3]
    loc_pc2 = pc2[..., :3]

    pc1_tiled = tf.tile(loc_pc1, (1, num_points_2, 1, 1))
    pc1_tiled = tf.reshape(pc1_tiled, [batch_size, num_points_2, num_points_1, 3])

    pc2_tiled = tf.tile(loc_pc2, (1, num_points_1, 1, 1))
    pc2_tiled = tf.reshape(pc2_tiled, [batch_size, num_points_1, num_points_2, 3])

    pc1_tiled_t = tf.transpose(pc1_tiled, perm=[0, 2, 1, 3])

    dist = pc1_tiled_t - pc2_tiled
    dist_squared = tf.square(dist)
    dist_squared_sum = tf.reduce_sum(dist_squared, axis=-1, keepdims=True)
    adj = tf.sqrt(dist_squared_sum)

    # For each poin in 'pc1' get the closest point in 'pc2'
    neg_adj = -adj

    min_idx = tf.argmax(neg_adj, axis=-2)

    idx_ = tf.range(batch_size, dtype=tf.int64) * num_points_2
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])
    pc2_flat = tf.reshape(pc2, [-1, 6])

    nearest_neighbours = tf.gather(pc2_flat, min_idx + idx_, axis=0)

    # For each pair of points, calculate the squared error.
    error = tf.squeeze(pc1) - tf.squeeze(nearest_neighbours)
    error = error[..., :3]
    error = tf.reshape(error, (-1, 1, 3))

    normal = tf.reshape(normal, (-1, 3, 1))

    projected_error = tf.square(tf.matmul(error, normal))
    projected_error = tf.reshape(projected_error, (batch_size, -1 ))


    return tf.reduce_mean(projected_error, axis=-1, keepdims=True)


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024*2,6))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
