from __future__ import division, print_function, absolute_import
from keras.models import Model, Sequential

from keras.layers import Input, Dropout, TimeDistributed, Masking, Dense, Conv3D, Conv3DTranspose, Reshape, ZeroPadding3D,\
    BatchNormalization, Embedding, Activation, LeakyReLU, MaxPooling3D, Multiply, Lambda, Flatten, Concatenate, Add, Maximum, \
    AveragePooling3D, GlobalAveragePooling3D, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, add, Conv2DTranspose, ConvLSTM2D

#from keras.layers.merge import Add, Concatenate
from keras.layers.recurrent import LSTM, GRU

#from keras.layers.normalization import BatchNormalization

import random
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils, plot_model, multi_gpu_model
from keras.utils.data_utils import get_file

from keras.regularizers import l2
import scipy.io as sio
import time
import datetime
import numpy as np
from numpy.random import randint
import json
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, TerminateOnNaN, CSVLogger, TensorBoard
from keras.optimizers import RMSprop, SGD, Adadelta
import sys

import tensorflow as tf
import tensorflow.image as tfi
import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array
import h5py

import os
import matplotlib.pyplot as plt
import cv2
from math import pi
import warnings
from keras.applications.vgg16 import VGG16, preprocess_input
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.resnet50 import ResNet50

from keras import initializers
from keras.engine import Layer, InputSpec

import argparse

import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from keras.utils import Sequence
from PIL import Image
from scipy.misc import imresize

os.sys.path.append('../ssd_keras')
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from bounding_box_utils.bounding_box_utils import iou, convert_coordinates
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy, match_multi


WEIGHTS_PATH_TH = 'https://dl.dropboxusercontent.com/s/rrp56zm347fbrdn/resnet101_weights_th.h5?dl=0'
WEIGHTS_PATH_TF = 'https://dl.dropboxusercontent.com/s/a21lyqwgf88nz9b/resnet101_weights_tf.h5?dl=0'
MD5_HASH_TH = '3d2e9a49d05192ce6e22200324b7defe'
MD5_HASH_TF = '867a922efc475e9966d0f3f7b884dc15'
# COMMON_MASK_W = 4 #modify this if maskW changes


I3D_WEIGHTS_NAME = ['rgb_kinetics_only', 'flow_kinetics_only', 'rgb_imagenet_and_kinetics',
                    'flow_imagenet_and_kinetics']

# path to pretrained models with top (classification layer)
I3D_WEIGHTS_PATH = {
    'rgb_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'flow_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'rgb_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
    'flow_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
}

# path to pretrained models with no top (no classification layer)
I3D_WEIGHTS_PATH_NO_TOP = {
    'rgb_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_kinetics_only': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'rgb_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_imagenet_and_kinetics': 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
}


class MultiGPUCheckpointCallback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)


def computeIOU(box1, box2):

    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]

    topLeftX1 = x1 - w1 / 2
    topLeftY1 = y1 - h1 / 2
    botRightX1 = x1 + w1 / 2
    botRightY1 = y1 + h1 / 2

    topLeftX2 = x2 - w2 / 2
    topLeftY2 = y2 - h2 / 2
    botRightX2 = x2 + w2 / 2
    botRightY2 = y2 + h2 / 2

    intersectionTLX = max(topLeftX1, topLeftX2)
    intersectionTLY = max(topLeftY1, topLeftY2)
    intersectionBRX = min(botRightX1, botRightX2)
    intersectionBRY = min(botRightY1, botRightY2)

    interArea = max(0, (intersectionBRX - intersectionTLX)) * max(0, (intersectionBRY - intersectionTLY))
    unionArea = w1 * h1 + w2 * h2 - interArea

    return interArea / unionArea


def save_h5_data(file_path, data_name, data):
    f = h5py.File(file_path, 'w')
    f.create_dataset(data_name, data=data)
    f.close()


def i3d_obtain_input_shape(input_shape,
                           default_frame_size,
                           min_frame_size,
                           default_num_frames,
                           min_num_frames,
                           data_format,
                           require_flatten,
                           weights=None):
    """Internal utility to compute/validate the model's input shape.
    (Adapted from `keras/applications/imagenet_utils.py`)

    # Arguments
        input_shape: either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_frame_size: default input frames(images) width/height for the model.
        min_frame_size: minimum input frames(images) width/height accepted by the model.
        default_num_frames: default input number of frames(images) for the model.
        min_num_frames: minimum input number of frames accepted by the model.
        data_format: image data format to use.
        require_flatten: whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: one of `None` (random initialization)
            or 'kinetics_only' (pre-training on Kinetics dataset).
            or 'imagenet_and_kinetics' (pre-training on ImageNet and Kinetics datasets).
            If weights='kinetics_only' or weights=='imagenet_and_kinetics' then
            input channels must be equal to 3.

    # Returns
        An integer shape tuple (may include None entries).

    # Raises
        ValueError: in case of invalid argument values.
    """
    if weights != 'kinetics_only' and weights != 'imagenet_and_kinetics' and input_shape and len(input_shape) == 4:
        if data_format == 'channels_first':
            if input_shape[0] not in {2, 3}:
                warnings.warn(
                    'This model usually expects 2 (for optical flow stream) or 3 (for RGB stream) input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_num_frames, default_frame_size, default_frame_size)
        else:
            if input_shape[-1] not in {2, 3}:
                warnings.warn(
                    'This model usually expects 2 (for optical flow stream) or 3 (for RGB stream) input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_num_frames, default_frame_size, default_frame_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_num_frames, default_frame_size, default_frame_size)
        else:
            default_shape = (default_num_frames, default_frame_size, default_frame_size, 3)
    if (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics') and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape

    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')
                if input_shape[0] != 3 and (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics'):
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if input_shape[1] is not None and input_shape[1] < min_num_frames:
                    raise ValueError('Input number of frames must be at least ' +
                                     str(min_num_frames) + '; got '
                                                           '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[2] is not None and input_shape[2] < min_frame_size) or
                        (input_shape[3] is not None and input_shape[3] < min_frame_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_frame_size) + 'x' + str(min_frame_size) + '; got '
                                                                                       '`input_shape=' + str(
                        input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')
                if input_shape[-1] != 3 and (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics'):
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if input_shape[0] is not None and input_shape[0] < min_num_frames:
                    raise ValueError('Input number of frames must be at least ' +
                                     str(min_num_frames) + '; got '
                                                           '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[1] is not None and input_shape[1] < min_frame_size) or
                        (input_shape[2] is not None and input_shape[2] < min_frame_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_frame_size) + 'x' + str(min_frame_size) + '; got '
                                                                                       '`input_shape=' + str(
                        input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None, None)
            else:
                input_shape = (None, None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


def conv3d_bn(x,
              filters,
              num_frames,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1, 1),
              use_bias=False,
              use_activation_fn=True,
              use_bn=True,
              bn_momentum=0.99,
              l2_reg=0.00005,
              name=None):
    """Utility function to apply conv3d + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_frames: frames (time depth) of the convolution kernel.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        use_bias: use bias or not
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
        l2_reg: l2 regularizer

    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv3D(
        filters, (num_frames, num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        kernel_regularizer=l2(l2_reg),
        name=conv_name)(x)

    if use_bn:
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 4
        x = BatchNormalization(axis=bn_axis, scale=False, momentum=bn_momentum, name=bn_name)(x)

    if use_activation_fn:
        x = Activation('relu', name=name)(x)

    return x


def cosineSimi(tensors):  # compute cosine similarity
    tiledTensor0 = K.tile(K.expand_dims(tensors[0], axis=1), [1, K.int_shape(tensors[1])[1], 1])
    return K.expand_dims(K.sum(tiledTensor0 * tensors[1], axis=-1) / (
            K.sqrt(K.sum(tiledTensor0 * tiledTensor0, axis=-1)) * K.sqrt(K.sum(tensors[1] * tensors[1], axis=-1))), axis=-1)


def rescaleTensor1(tensor):  # rescale tensor to [-1,1]
    tensor = K.squeeze(tensor, axis=-1)
    maxes = K.max(tensor, axis=-1, keepdims=True)
    mins = K.min(tensor, axis=-1, keepdims=True)
    meanSubbed = tensor - K.tile((maxes + mins) / 2., (1, 8732))
    meanSubbedMaxes = K.max(meanSubbed, axis=-1, keepdims=True)
    rescaled = meanSubbed / K.tile(meanSubbedMaxes, (1, 8732))
    return K.expand_dims(rescaled, axis=-1)


def rescaleTensor2(tensor):  # rescale tensor to [-1,1]
    #tensor = K.squeeze(tensor, axis=-1)
    maxes = K.max(tensor, axis=-1, keepdims=True)
    mins = K.min(tensor, axis=-1, keepdims=True)
    meanSubbed = tensor - K.tile((maxes + mins) / 2., (1, NUM_CLASS))
    meanSubbedMaxes = K.max(meanSubbed, axis=-1, keepdims=True)
    rescaled = meanSubbed / K.tile(meanSubbedMaxes, (1, NUM_CLASS))
    return rescaled


def softPredBoxClassification(tensors):  # select the box which has the highest attention logits for each batch (1 image + 1 sequence) and return its classification logits
    probs = K.softmax(K.squeeze(tensors[0], axis=-1))
    tiledProbs = K.tile(K.expand_dims(probs, axis=-1), [1, 1, NUM_CLASS])
    return K.sum(tiledProbs * tensors[1], axis=1) #probs weighted classification score


def predBoxClassification(
        tensors):  # select the box which has the highest attention logits for each batch (1 image + 1 sequence) and return its classification logits
    col = K.argmax(K.squeeze(tensors[0], axis=-1))
    row = tf.range(tf.shape(col)[0])
    idx = tf.stack([tf.cast(row, col.dtype), col], axis=1)
    return tf.gather_nd(tensors[1], idx, name='tf_gather_pred')


def identity_layer(tensor):
    return tensor


def squeezeLayer(tensor):
    return K.squeeze(tensor, axis=-1)


def removeTime(tensor):
    return K.squeeze(tensor, axis=1)


def my_zeros_like(tensor):
    return K.zeros_like(tensor)


def mindReader(image_size,
               sequence_LHW,
               n_classes=24,
               mode='training',
               l2_regularization=0.00005,
               min_scale=None,
               max_scale=None,
               scales=[0.07, 0.15, 0.32, 0.49, 0.66, 0.83, 1.0],
               aspect_ratios_global=None,
               aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                        [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                        [1.0, 2.0, 0.5],
                                        [1.0, 2.0, 0.5]],
               two_boxes_for_ar1=True,
               steps=[8, 16, 32, 64, 100, 300],
               offsets=None,
               clip_boxes=False,
               variances=[0.1, 0.1, 0.2, 0.2],
               coords='centroids',
               normalize_coords=True,
               subtract_mean=[123, 117, 104],
               divide_by_stddev=None,
               swap_channels=[2, 1, 0],
               vgg_useBN=False,
               vgg_BN_momentum=0.99,
               i3d_useBN=False,
               i3d_BN_momentum=0.99,
               whereToHelpWhat=True,
               whatToHelpWhere=True,
               whereHelpInside=True,
               whatHelpInside=True,
               useRGBStream=True,
               useFlowStream=True,
               temporal_channels=[256, 128],
               softArgmax=True
               ):
    '''
      Parts of the code are inspired by: I3D --- https://github.com/dlpbc/keras-kinetics-i3d, https://arxiv.org/abs/1705.07750
                                         SSD --- https://github.com/pierluigiferrari/ssd_keras, https://arxiv.org/abs/1512.02325
    '''

    n_predictor_layers = 6

    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)
            else:
                n_boxes.append(len(ar))
    else:
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                            tensor[..., swap_channels[3]]], axis=-1)

    l2_reg = l2_regularization
    sequence_length, sequence_height, sequence_width = sequence_LHW

    ### RGB stream
    if useRGBStream:

        rgb_input = Input(shape=(sequence_length, sequence_height, sequence_width, 3))

        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 4

        # Downsampling via convolution (spatial and temporal)
        rgb_x = conv3d_bn(rgb_input, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                          bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_1a_7x7')

        # Downsampling (spatial only)
        rgb_x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='rgb_MaxPool2d_2a_3x3')(rgb_x)
        rgb_x = conv3d_bn(rgb_x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                          bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_2b_1x1')
        rgb_x = conv3d_bn(rgb_x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                          bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_2c_3x3')

        # Downsampling (spatial only)
        rgb_x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='rgb_MaxPool2d_3a_3x3')(rgb_x)

        # Mixed 3b
        rgb_branch_0 = conv3d_bn(rgb_x, 64, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3b_0a_1x1')

        rgb_branch_1 = conv3d_bn(rgb_x, 96, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3b_1a_1x1')
        rgb_branch_1 = conv3d_bn(rgb_branch_1, 128, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3b_1b_3x3')

        rgb_branch_2 = conv3d_bn(rgb_x, 16, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3b_2a_1x1')
        rgb_branch_2 = conv3d_bn(rgb_branch_2, 32, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3b_2b_3x3')

        rgb_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='rgb_MaxPool2d_3b_3a_3x3')(rgb_x)
        rgb_branch_3 = conv3d_bn(rgb_branch_3, 32, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3b_3b_1x1')

        rgb_x = Concatenate(
            axis=channel_axis,
            name='rgb_Mixed_3b')([rgb_branch_0, rgb_branch_1, rgb_branch_2, rgb_branch_3])

        # Mixed 3c
        rgb_branch_0 = conv3d_bn(rgb_x, 128, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3c_0a_1x1')

        rgb_branch_1 = conv3d_bn(rgb_x, 128, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3c_1a_1x1')
        rgb_branch_1 = conv3d_bn(rgb_branch_1, 192, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3c_1b_3x3')

        rgb_branch_2 = conv3d_bn(rgb_x, 32, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3c_2a_1x1')
        rgb_branch_2 = conv3d_bn(rgb_branch_2, 96, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3c_2b_3x3')

        rgb_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='rgb_MaxPool2d_3c_3a_3x3')(rgb_x)
        rgb_branch_3 = conv3d_bn(rgb_branch_3, 64, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_3c_3b_1x1')

        rgb_x_3c = Concatenate(
            axis=channel_axis,
            name='rgb_Mixed_3c')([rgb_branch_0, rgb_branch_1, rgb_branch_2, rgb_branch_3])

        # Downsampling (spatial and temporal)
        rgb_x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='rgb_MaxPool2d_4a_3x3')(rgb_x_3c)

        # Mixed 4b
        rgb_branch_0 = conv3d_bn(rgb_x, 192, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4b_0a_1x1')

        rgb_branch_1 = conv3d_bn(rgb_x, 96, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4b_1a_1x1')
        rgb_branch_1 = conv3d_bn(rgb_branch_1, 208, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4b_1b_3x3')

        rgb_branch_2 = conv3d_bn(rgb_x, 16, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4b_2a_1x1')
        rgb_branch_2 = conv3d_bn(rgb_branch_2, 48, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4b_2b_3x3')

        rgb_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='rgb_MaxPool2d_4b_3a_3x3')(rgb_x)
        rgb_branch_3 = conv3d_bn(rgb_branch_3, 64, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4b_3b_1x1')

        rgb_x = Concatenate(
            axis=channel_axis,
            name='rgb_Mixed_4b')([rgb_branch_0, rgb_branch_1, rgb_branch_2, rgb_branch_3])

        # Mixed 4c
        rgb_branch_0 = conv3d_bn(rgb_x, 160, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4c_0a_1x1')

        rgb_branch_1 = conv3d_bn(rgb_x, 112, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4c_1a_1x1')
        rgb_branch_1 = conv3d_bn(rgb_branch_1, 224, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4c_1b_3x3')

        rgb_branch_2 = conv3d_bn(rgb_x, 24, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4c_2a_1x1')
        rgb_branch_2 = conv3d_bn(rgb_branch_2, 64, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4c_2b_3x3')

        rgb_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='rgb_MaxPool2d_4c_3a_3x3')(rgb_x)
        rgb_branch_3 = conv3d_bn(rgb_branch_3, 64, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4c_3b_1x1')

        rgb_x = Concatenate(
            axis=channel_axis,
            name='rgb_Mixed_4c')([rgb_branch_0, rgb_branch_1, rgb_branch_2, rgb_branch_3])

        # Mixed 4d
        rgb_branch_0 = conv3d_bn(rgb_x, 128, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4d_0a_1x1')

        rgb_branch_1 = conv3d_bn(rgb_x, 128, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4d_1a_1x1')
        rgb_branch_1 = conv3d_bn(rgb_branch_1, 256, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4d_1b_3x3')

        rgb_branch_2 = conv3d_bn(rgb_x, 24, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4d_2a_1x1')
        rgb_branch_2 = conv3d_bn(rgb_branch_2, 64, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4d_2b_3x3')

        rgb_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='rgb_MaxPool2d_4d_3a_3x3')(rgb_x)
        rgb_branch_3 = conv3d_bn(rgb_branch_3, 64, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4d_3b_1x1')

        rgb_x = Concatenate(
            axis=channel_axis,
            name='rgb_Mixed_4d')([rgb_branch_0, rgb_branch_1, rgb_branch_2, rgb_branch_3])

        # Mixed 4e
        rgb_branch_0 = conv3d_bn(rgb_x, 112, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4e_0a_1x1')

        rgb_branch_1 = conv3d_bn(rgb_x, 144, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4e_1a_1x1')
        rgb_branch_1 = conv3d_bn(rgb_branch_1, 288, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4e_1b_3x3')

        rgb_branch_2 = conv3d_bn(rgb_x, 32, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4e_2a_1x1')
        rgb_branch_2 = conv3d_bn(rgb_branch_2, 64, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4e_2b_3x3')

        rgb_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='rgb_MaxPool2d_4e_3a_3x3')(rgb_x)
        rgb_branch_3 = conv3d_bn(rgb_branch_3, 64, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4e_3b_1x1')

        rgb_x = Concatenate(
            axis=channel_axis,
            name='rgb_Mixed_4e')([rgb_branch_0, rgb_branch_1, rgb_branch_2, rgb_branch_3])

        # Mixed 4f
        rgb_branch_0 = conv3d_bn(rgb_x, 256, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4f_0a_1x1')

        rgb_branch_1 = conv3d_bn(rgb_x, 160, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4f_1a_1x1')
        rgb_branch_1 = conv3d_bn(rgb_branch_1, 320, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4f_1b_3x3')

        rgb_branch_2 = conv3d_bn(rgb_x, 32, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4f_2a_1x1')
        rgb_branch_2 = conv3d_bn(rgb_branch_2, 128, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4f_2b_3x3')

        rgb_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='rgb_MaxPool2d_4f_3a_3x3')(rgb_x)
        rgb_branch_3 = conv3d_bn(rgb_branch_3, 128, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_4f_3b_1x1')

        rgb_x_4f = Concatenate(
            axis=channel_axis,
            name='rgb_Mixed_4f')([rgb_branch_0, rgb_branch_1, rgb_branch_2, rgb_branch_3])

        # Downsampling (spatial and temporal)
        rgb_x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='rgb_MaxPool2d_5a_2x2')(rgb_x_4f)

        # Mixed 5b
        rgb_branch_0 = conv3d_bn(rgb_x, 256, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5b_0a_1x1')

        rgb_branch_1 = conv3d_bn(rgb_x, 160, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5b_1a_1x1')
        rgb_branch_1 = conv3d_bn(rgb_branch_1, 320, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5b_1b_3x3')

        rgb_branch_2 = conv3d_bn(rgb_x, 32, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5b_2a_1x1')
        rgb_branch_2 = conv3d_bn(rgb_branch_2, 128, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5b_2b_3x3')

        rgb_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='rgb_MaxPool2d_5b_3a_3x3')(rgb_x)
        rgb_branch_3 = conv3d_bn(rgb_branch_3, 128, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5b_3b_1x1')

        rgb_x = Concatenate(
            axis=channel_axis,
            name='rgb_Mixed_5b')([rgb_branch_0, rgb_branch_1, rgb_branch_2, rgb_branch_3])

        # Mixed 5c
        rgb_branch_0 = conv3d_bn(rgb_x, 384, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5c_0a_1x1')

        rgb_branch_1 = conv3d_bn(rgb_x, 192, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5c_1a_1x1')
        rgb_branch_1 = conv3d_bn(rgb_branch_1, 384, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5c_1b_3x3')

        rgb_branch_2 = conv3d_bn(rgb_x, 48, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5c_2a_1x1')
        rgb_branch_2 = conv3d_bn(rgb_branch_2, 128, 3, 3, 3, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5c_2b_3x3')

        rgb_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='rgb_MaxPool2d_5c_3a_3x3')(rgb_x)
        rgb_branch_3 = conv3d_bn(rgb_branch_3, 128, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                 bn_momentum=i3d_BN_momentum, name='rgb_Conv3d_5c_3b_1x1')

        rgb_x = Concatenate(
            axis=channel_axis,
            name='rgb_Mixed_5c')([rgb_branch_0, rgb_branch_1, rgb_branch_2, rgb_branch_3])

    ### Flow steam
    if useFlowStream:

        flow_input = Input(shape=(sequence_length, sequence_height, sequence_width, 2))

        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 4

        # Downsampling via convolution (spatial and temporal)
        flow_x = conv3d_bn(flow_input, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', l2_reg=l2_reg,
                           use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_1a_7x7')

        # Downsampling (spatial only)
        flow_x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='flow_MaxPool2d_2a_3x3')(flow_x)
        flow_x = conv3d_bn(flow_x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', l2_reg=l2_reg,
                           use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_2b_1x1')
        flow_x = conv3d_bn(flow_x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', l2_reg=l2_reg,
                           use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_2c_3x3')

        # Downsampling (spatial only)
        flow_x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='flow_MaxPool2d_3a_3x3')(flow_x)

        # Mixed 3b
        flow_branch_0 = conv3d_bn(flow_x, 64, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3b_0a_1x1')

        flow_branch_1 = conv3d_bn(flow_x, 96, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3b_1a_1x1')
        flow_branch_1 = conv3d_bn(flow_branch_1, 128, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3b_1b_3x3')

        flow_branch_2 = conv3d_bn(flow_x, 16, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3b_2a_1x1')
        flow_branch_2 = conv3d_bn(flow_branch_2, 32, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3b_2b_3x3')

        flow_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='flow_MaxPool2d_3b_3a_3x3')(
            flow_x)
        flow_branch_3 = conv3d_bn(flow_branch_3, 32, 1, 1, 1, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3b_3b_1x1')

        flow_x = Concatenate(
            axis=channel_axis,
            name='flow_Mixed_3b')([flow_branch_0, flow_branch_1, flow_branch_2, flow_branch_3])

        # Mixed 3c
        flow_branch_0 = conv3d_bn(flow_x, 128, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3c_0a_1x1')

        flow_branch_1 = conv3d_bn(flow_x, 128, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3c_1a_1x1')
        flow_branch_1 = conv3d_bn(flow_branch_1, 192, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3c_1b_3x3')

        flow_branch_2 = conv3d_bn(flow_x, 32, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3c_2a_1x1')
        flow_branch_2 = conv3d_bn(flow_branch_2, 96, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3c_2b_3x3')

        flow_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='flow_MaxPool2d_3c_3a_3x3')(
            flow_x)
        flow_branch_3 = conv3d_bn(flow_branch_3, 64, 1, 1, 1, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_3c_3b_1x1')

        flow_x_3c = Concatenate(
            axis=channel_axis,
            name='flow_Mixed_3c')([flow_branch_0, flow_branch_1, flow_branch_2, flow_branch_3])

        # Downsampling (spatial and temporal)
        flow_x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='flow_MaxPool2d_4a_3x3')(flow_x_3c)

        # Mixed 4b
        flow_branch_0 = conv3d_bn(flow_x, 192, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4b_0a_1x1')

        flow_branch_1 = conv3d_bn(flow_x, 96, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4b_1a_1x1')
        flow_branch_1 = conv3d_bn(flow_branch_1, 208, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4b_1b_3x3')

        flow_branch_2 = conv3d_bn(flow_x, 16, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4b_2a_1x1')
        flow_branch_2 = conv3d_bn(flow_branch_2, 48, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4b_2b_3x3')

        flow_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='flow_MaxPool2d_4b_3a_3x3')(
            flow_x)
        flow_branch_3 = conv3d_bn(flow_branch_3, 64, 1, 1, 1, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4b_3b_1x1')

        flow_x = Concatenate(
            axis=channel_axis,
            name='flow_Mixed_4b')([flow_branch_0, flow_branch_1, flow_branch_2, flow_branch_3])

        # Mixed 4c
        flow_branch_0 = conv3d_bn(flow_x, 160, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4c_0a_1x1')

        flow_branch_1 = conv3d_bn(flow_x, 112, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4c_1a_1x1')
        flow_branch_1 = conv3d_bn(flow_branch_1, 224, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4c_1b_3x3')

        flow_branch_2 = conv3d_bn(flow_x, 24, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4c_2a_1x1')
        flow_branch_2 = conv3d_bn(flow_branch_2, 64, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4c_2b_3x3')

        flow_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='flow_MaxPool2d_4c_3a_3x3')(
            flow_x)
        flow_branch_3 = conv3d_bn(flow_branch_3, 64, 1, 1, 1, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4c_3b_1x1')

        flow_x = Concatenate(
            axis=channel_axis,
            name='flow_Mixed_4c')([flow_branch_0, flow_branch_1, flow_branch_2, flow_branch_3])

        # Mixed 4d
        flow_branch_0 = conv3d_bn(flow_x, 128, 1, 1, 1, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4d_0a_1x1')

        flow_branch_1 = conv3d_bn(flow_x, 128, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4d_1a_1x1')
        flow_branch_1 = conv3d_bn(flow_branch_1, 256, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4d_1b_3x3')

        flow_branch_2 = conv3d_bn(flow_x, 24, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4d_2a_1x1')
        flow_branch_2 = conv3d_bn(flow_branch_2, 64, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4d_2b_3x3')

        flow_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='flow_MaxPool2d_4d_3a_3x3')(
            flow_x)
        flow_branch_3 = conv3d_bn(flow_branch_3, 64, 1, 1, 1, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4d_3b_1x1')

        flow_x = Concatenate(
            axis=channel_axis,
            name='flow_Mixed_4d')([flow_branch_0, flow_branch_1, flow_branch_2, flow_branch_3])

        # Mixed 4e
        flow_branch_0 = conv3d_bn(flow_x, 112, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4e_0a_1x1')

        flow_branch_1 = conv3d_bn(flow_x, 144, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4e_1a_1x1')
        flow_branch_1 = conv3d_bn(flow_branch_1, 288, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4e_1b_3x3')

        flow_branch_2 = conv3d_bn(flow_x, 32, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4e_2a_1x1')
        flow_branch_2 = conv3d_bn(flow_branch_2, 64, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4e_2b_3x3')

        flow_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='flow_MaxPool2d_4e_3a_3x3')(
            flow_x)
        flow_branch_3 = conv3d_bn(flow_branch_3, 64, 1, 1, 1, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4e_3b_1x1')

        flow_x = Concatenate(
            axis=channel_axis,
            name='flow_Mixed_4e')([flow_branch_0, flow_branch_1, flow_branch_2, flow_branch_3])

        # Mixed 4f
        flow_branch_0 = conv3d_bn(flow_x, 256, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4f_0a_1x1')

        flow_branch_1 = conv3d_bn(flow_x, 160, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4f_1a_1x1')
        flow_branch_1 = conv3d_bn(flow_branch_1, 320, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4f_1b_3x3')

        flow_branch_2 = conv3d_bn(flow_x, 32, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4f_2a_1x1')
        flow_branch_2 = conv3d_bn(flow_branch_2, 128, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4f_2b_3x3')

        flow_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='flow_MaxPool2d_4f_3a_3x3')(
            flow_x)
        flow_branch_3 = conv3d_bn(flow_branch_3, 128, 1, 1, 1, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_4f_3b_1x1')

        flow_x_4f = Concatenate(
            axis=channel_axis,
            name='flow_Mixed_4f')([flow_branch_0, flow_branch_1, flow_branch_2, flow_branch_3])

        # Downsampling (spatial and temporal)
        flow_x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='flow_MaxPool2d_5a_2x2')(flow_x_4f)

        # Mixed 5b
        flow_branch_0 = conv3d_bn(flow_x, 256, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5b_0a_1x1')

        flow_branch_1 = conv3d_bn(flow_x, 160, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5b_1a_1x1')
        flow_branch_1 = conv3d_bn(flow_branch_1, 320, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5b_1b_3x3')

        flow_branch_2 = conv3d_bn(flow_x, 32, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5b_2a_1x1')
        flow_branch_2 = conv3d_bn(flow_branch_2, 128, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5b_2b_3x3')

        flow_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='flow_MaxPool2d_5b_3a_3x3')(
            flow_x)
        flow_branch_3 = conv3d_bn(flow_branch_3, 128, 1, 1, 1, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5b_3b_1x1')

        flow_x = Concatenate(
            axis=channel_axis,
            name='flow_Mixed_5b')([flow_branch_0, flow_branch_1, flow_branch_2, flow_branch_3])

        # Mixed 5c
        flow_branch_0 = conv3d_bn(flow_x, 384, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5c_0a_1x1')

        flow_branch_1 = conv3d_bn(flow_x, 192, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5c_1a_1x1')
        flow_branch_1 = conv3d_bn(flow_branch_1, 384, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5c_1b_3x3')

        flow_branch_2 = conv3d_bn(flow_x, 48, 1, 1, 1, padding='same', l2_reg=l2_reg, use_bn=i3d_useBN,
                                  bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5c_2a_1x1')
        flow_branch_2 = conv3d_bn(flow_branch_2, 128, 3, 3, 3, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5c_2b_3x3')

        flow_branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='flow_MaxPool2d_5c_3a_3x3')(
            flow_x)
        flow_branch_3 = conv3d_bn(flow_branch_3, 128, 1, 1, 1, padding='same', l2_reg=l2_reg,
                                  use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum, name='flow_Conv3d_5c_3b_1x1')

        flow_x = Concatenate(
            axis=channel_axis,
            name='flow_Mixed_5c')([flow_branch_0, flow_branch_1, flow_branch_2, flow_branch_3])


    if useRGBStream and useFlowStream:

        attention_x_3c = Add(name='fuse_attention_3c_add')([rgb_x_3c, flow_x_3c])
        attention_x_4f = Add(name='fuse_attention_4f_add')([rgb_x_4f, flow_x_4f])
        attention_x = Add(name='fuse_attention_x_add')([rgb_x, flow_x])

        attention_x_3c_feat = conv3d_bn(attention_x_3c, 1024, K.int_shape(attention_x_3c)[1], 1, 1,
                                        padding='valid',
                                        strides=(K.int_shape(attention_x_3c)[1], 1, 1),
                                        l2_reg=l2_reg, use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum,
                                        name='attention_x_3c_feat')

        attention_x_3c_feat = Lambda(removeTime, name='attention_x_3c_feat_squeeze')(attention_x_3c_feat)

        attention_x_4f_feat = conv3d_bn(attention_x_4f, 1024, int(attention_x_4f.shape[1]), 1, 1,
                                        padding='valid',
                                        strides=(int(attention_x_4f.shape[1]), 1, 1),
                                        l2_reg=l2_reg, use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum,
                                        name='attention_x_4f_feat')
        attention_x_4f_feat = Lambda(removeTime, name='attention_x_4f_feat_squeeze')(attention_x_4f_feat)

        attention_x_feat = conv3d_bn(attention_x, 1024, int(attention_x.shape[1]), 1, 1, padding='valid',
                                     strides=(int(attention_x.shape[1]), 1, 1),
                                     l2_reg=l2_reg, use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum,
                                     name='attention_x_feat')
        attention_x_feat = Lambda(removeTime, name='attention_x_feat_squeeze')(attention_x_feat)

        attention_x_6_1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg),
                                 name='attention_x_6_1')(attention_x_feat)
        if vgg_useBN: attention_x_6_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_6_1')(attention_x_6_1)
        attention_x_6_1 = Activation('relu', name='attention_x_6_1_relu')(attention_x_6_1)
        attention_x_6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='attention_x_6_1_padding')(
            attention_x_6_1)
        attention_x_6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg), name='attention_x_6_2')(attention_x_6_1)
        if vgg_useBN: attention_x_6_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_6_2')(attention_x_6_2)
        attention_x_6_2 = Activation('relu', name='attention_x_6_2_relu')(attention_x_6_2)

        attention_x_7_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg),
                                 name='attention_x_7_1')(attention_x_6_2)
        if vgg_useBN: attention_x_7_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_7_1')(attention_x_7_1)
        attention_x_7_1 = Activation('relu', name='attention_x_7_1_relu')(attention_x_7_1)
        attention_x_7_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg), name='attention_x_7_2')(attention_x_7_1)
        if vgg_useBN: attention_x_7_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_7_2')(
            attention_x_7_2)
        attention_x_7_2 = Activation('relu', name='attention_x_7_2_relu')(attention_x_7_2)

        attention_x_8_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg),
                                 name='attention_x_8_1')(attention_x_7_2)
        if vgg_useBN: attention_x_8_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_8_1')(attention_x_8_1)
        attention_x_8_1 = Activation('relu', name='attention_x_8_1_relu')(attention_x_8_1)
        attention_x_8_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg), name='attention_x_8_2')(attention_x_8_1)
        if vgg_useBN: attention_x_8_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_8_2')(attention_x_8_2)
        attention_x_8_2 = Activation('relu', name='attention_x_8_2_relu')(attention_x_8_2)

        attention_x_3c_attn = Conv2D(n_boxes[0], (3, 3), padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=l2(l2_reg), name='attention_x_3c_attn')(
            attention_x_3c_feat)
        attention_x_4f_attn = Conv2D(n_boxes[1], (3, 3), padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=l2(l2_reg), name='attention_x_4f_attn')(
            attention_x_4f_feat)
        attention_x_attn = Conv2D(n_boxes[2], (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='attention_x_attn')(attention_x_feat)
        attention_x_6_attn = Conv2D(n_boxes[3], (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='attention_x_6_attn')(
            attention_x_6_2)
        attention_x_7_attn = Conv2D(n_boxes[4], (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='attention_x_7_attn')(
            attention_x_7_2)
        attention_x_8_attn = Conv2D(n_boxes[5], (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='attention_x_8_attn')(
            attention_x_8_2)

        attention_x_3c_attn_reshape = Reshape((-1,), name='attention_x_3c_attn_reshape')(
            attention_x_3c_attn)
        attention_x_4f_attn_reshape = Reshape((-1,), name='attention_x_4f_attn_reshape')(
            attention_x_4f_attn)
        attention_x_attn_reshape = Reshape((-1,), name='attention_x_attn_reshape')(attention_x_attn)
        attention_x_6_attn_reshape = Reshape((-1,), name='attention_x_6_attn_reshape')(attention_x_6_attn)
        attention_x_7_attn_reshape = Reshape((-1,), name='attention_x_7_attn_reshape')(attention_x_7_attn)
        attention_x_8_attn_reshape = Reshape((-1,), name='attention_x_8_attn_reshape')(attention_x_8_attn)

        attention_logits = Concatenate(axis=1, name='attention_attention')([attention_x_3c_attn_reshape,
                                                                            attention_x_4f_attn_reshape,
                                                                            attention_x_attn_reshape,
                                                                            attention_x_6_attn_reshape,
                                                                            attention_x_7_attn_reshape,
                                                                            attention_x_8_attn_reshape])

        fused_block5 = Add(name='fuse_i3d_block5_add')([rgb_x, flow_x])

        l = int(fused_block5.shape[1])
        h = int(fused_block5.shape[2])
        w = int(fused_block5.shape[3])

        temporal_embedding = conv3d_bn(fused_block5, temporal_channels[0], l, 1, 1, padding='valid',
                                       strides=(l, 1, 1), l2_reg=l2_reg,
                                       name='i3d_temporal_embedding_1')
        temporal_embedding = conv3d_bn(temporal_embedding, temporal_channels[1], 1, 2, 2, padding='same',
                                       strides=(1, 2, 2), l2_reg=l2_reg,
                                       name='i3d_temporal_embedding_2')

        temporal_embedding = Flatten(name='i3d_to_vec')(temporal_embedding)

        if whatToHelpWhere:

            i3d_toy_logits = Dense(n_classes, activation='linear', kernel_regularizer=l2(l2_reg),
                                   name='i3d_toy_logits')(temporal_embedding)

    elif useRGBStream:

        attention_x_3c = Lambda(identity_layer, name='attention_x_3c_identity')(rgb_x_3c)
        attention_x_4f = Lambda(identity_layer, name='attention_x_4f_identity')(rgb_x_4f)
        attention_x = Lambda(identity_layer, name='attention_x_identity')(rgb_x)

        attention_x_3c_feat = conv3d_bn(attention_x_3c, 1024, K.int_shape(attention_x_3c)[1], 1, 1,
                                        padding='valid',
                                        strides=(K.int_shape(attention_x_3c)[1], 1, 1),
                                        l2_reg=l2_reg, use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum,
                                        name='attention_x_3c_feat')
        attention_x_3c_feat = Lambda(removeTime, name='attention_x_3c_feat_squeeze')(attention_x_3c_feat)

        attention_x_4f_feat = conv3d_bn(attention_x_4f, 1024, int(attention_x_4f.shape[1]), 1, 1,
                                        padding='valid',
                                        strides=(int(attention_x_4f.shape[1]), 1, 1),
                                        l2_reg=l2_reg, use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum,
                                        name='attention_x_4f_feat')
        attention_x_4f_feat = Lambda(removeTime, name='attention_x_4f_feat_squeeze')(attention_x_4f_feat)

        attention_x_feat = conv3d_bn(attention_x, 1024, int(attention_x.shape[1]), 1, 1, padding='valid',
                                     strides=(int(attention_x.shape[1]), 1, 1),
                                     l2_reg=l2_reg, use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum,
                                     name='attention_x_feat')
        attention_x_feat = Lambda(removeTime, name='attention_x_feat_squeeze')(attention_x_feat)

        attention_x_6_1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg),
                                 name='attention_x_6_1')(attention_x_feat)
        if vgg_useBN: attention_x_6_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_6_1')(attention_x_6_1)
        attention_x_6_1 = Activation('relu', name='attention_x_6_1_relu')(attention_x_6_1)
        attention_x_6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='attention_x_6_1_padding')(
            attention_x_6_1)
        attention_x_6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg), name='attention_x_6_2')(attention_x_6_1)
        if vgg_useBN: attention_x_6_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_6_2')(attention_x_6_2)
        attention_x_6_2 = Activation('relu', name='attention_x_6_2_relu')(attention_x_6_2)

        attention_x_7_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg),
                                 name='attention_x_7_1')(attention_x_6_2)
        if vgg_useBN: attention_x_7_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_7_1')(attention_x_7_1)
        attention_x_7_1 = Activation('relu', name='attention_x_7_1_relu')(attention_x_7_1)
        attention_x_7_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg), name='attention_x_7_2')(attention_x_7_1)
        if vgg_useBN: attention_x_7_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_7_2')(
            attention_x_7_2)
        attention_x_7_2 = Activation('relu', name='attention_x_7_2_relu')(attention_x_7_2)

        attention_x_8_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg),
                                 name='attention_x_8_1')(attention_x_7_2)
        if vgg_useBN: attention_x_8_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_8_1')(attention_x_8_1)
        attention_x_8_1 = Activation('relu', name='attention_x_8_1_relu')(attention_x_8_1)
        attention_x_8_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg), name='attention_x_8_2')(attention_x_8_1)
        if vgg_useBN: attention_x_8_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_8_2')(attention_x_8_2)
        attention_x_8_2 = Activation('relu', name='attention_x_8_2_relu')(attention_x_8_2)

        attention_x_3c_attn = Conv2D(n_boxes[0], (3, 3), padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=l2(l2_reg), name='attention_x_3c_attn')(
            attention_x_3c_feat)
        attention_x_4f_attn = Conv2D(n_boxes[1], (3, 3), padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=l2(l2_reg), name='attention_x_4f_attn')(
            attention_x_4f_feat)
        attention_x_attn = Conv2D(n_boxes[2], (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='attention_x_attn')(attention_x_feat)
        attention_x_6_attn = Conv2D(n_boxes[3], (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='attention_x_6_attn')(
            attention_x_6_2)
        attention_x_7_attn = Conv2D(n_boxes[4], (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='attention_x_7_attn')(
            attention_x_7_2)
        attention_x_8_attn = Conv2D(n_boxes[5], (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='attention_x_8_attn')(
            attention_x_8_2)

        attention_x_3c_attn_reshape = Reshape((-1,), name='attention_x_3c_attn_reshape')(
            attention_x_3c_attn)
        attention_x_4f_attn_reshape = Reshape((-1,), name='attention_x_4f_attn_reshape')(
            attention_x_4f_attn)
        attention_x_attn_reshape = Reshape((-1,), name='attention_x_attn_reshape')(attention_x_attn)
        attention_x_6_attn_reshape = Reshape((-1,), name='attention_x_6_attn_reshape')(attention_x_6_attn)
        attention_x_7_attn_reshape = Reshape((-1,), name='attention_x_7_attn_reshape')(attention_x_7_attn)
        attention_x_8_attn_reshape = Reshape((-1,), name='attention_x_8_attn_reshape')(attention_x_8_attn)

        attention_logits = Concatenate(axis=1, name='attention_attention')([attention_x_3c_attn_reshape,
                                                                            attention_x_4f_attn_reshape,
                                                                            attention_x_attn_reshape,
                                                                            attention_x_6_attn_reshape,
                                                                            attention_x_7_attn_reshape,
                                                                            attention_x_8_attn_reshape])


        l = int(rgb_x.shape[1])
        h = int(rgb_x.shape[2])
        w = int(rgb_x.shape[3])

        rgb_embedding = conv3d_bn(rgb_x, temporal_channels[0], l, 1, 1, strides=(l, 1, 1),
                                  padding='valid', l2_reg=l2_reg,
                                  name='rgb_temporal_embedding_1')
        rgb_embedding = conv3d_bn(rgb_embedding, temporal_channels[1], 1, 2, 2, strides=(1, 2, 2),
                                  padding='same', l2_reg=l2_reg,
                                  name='rgb_temporal_embedding_2')

        temporal_embedding = Flatten(name='rgb_to_vec')(rgb_embedding)

        if whatToHelpWhere:
            i3d_toy_logits = Dense(n_classes, activation='linear', kernel_regularizer=l2(l2_reg),
                                   name='i3d_toy_logits')(temporal_embedding)



    elif useFlowStream:

        attention_x_3c = Lambda(identity_layer, name='attention_x_3c_identity')(flow_x_3c)
        attention_x_4f = Lambda(identity_layer, name='attention_x_4f_identity')(flow_x_4f)
        attention_x = Lambda(identity_layer, name='attention_x_identity')(flow_x)

        attention_x_3c_feat = conv3d_bn(attention_x_3c, 1024, K.int_shape(attention_x_3c)[1], 1, 1,
                                        padding='valid',
                                        strides=(K.int_shape(attention_x_3c)[1], 1, 1),
                                        l2_reg=l2_reg, use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum,
                                        name='attention_x_3c_feat')
        attention_x_3c_feat = Lambda(removeTime, name='attention_x_3c_feat_squeeze')(attention_x_3c_feat)

        attention_x_4f_feat = conv3d_bn(attention_x_4f, 1024, int(attention_x_4f.shape[1]), 1, 1,
                                        padding='valid',
                                        strides=(int(attention_x_4f.shape[1]), 1, 1),
                                        l2_reg=l2_reg, use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum,
                                        name='attention_x_4f_feat')
        attention_x_4f_feat = Lambda(removeTime, name='attention_x_4f_feat_squeeze')(attention_x_4f_feat)

        attention_x_feat = conv3d_bn(attention_x, 1024, int(attention_x.shape[1]), 1, 1, padding='valid',
                                     strides=(int(attention_x.shape[1]), 1, 1),
                                     l2_reg=l2_reg, use_bn=i3d_useBN, bn_momentum=i3d_BN_momentum,
                                     name='attention_x_feat')
        attention_x_feat = Lambda(removeTime, name='attention_x_feat_squeeze')(attention_x_feat)

        attention_x_6_1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg),
                                 name='attention_x_6_1')(attention_x_feat)
        if vgg_useBN: attention_x_6_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_6_1')(attention_x_6_1)
        attention_x_6_1 = Activation('relu', name='attention_x_6_1_relu')(attention_x_6_1)
        attention_x_6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='attention_x_6_1_padding')(
            attention_x_6_1)
        attention_x_6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg), name='attention_x_6_2')(attention_x_6_1)
        if vgg_useBN: attention_x_6_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_6_2')(attention_x_6_2)
        attention_x_6_2 = Activation('relu', name='attention_x_6_2_relu')(attention_x_6_2)

        attention_x_7_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg),
                                 name='attention_x_7_1')(attention_x_6_2)
        if vgg_useBN: attention_x_7_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_7_1')(attention_x_7_1)
        attention_x_7_1 = Activation('relu', name='attention_x_7_1_relu')(attention_x_7_1)
        attention_x_7_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg), name='attention_x_7_2')(attention_x_7_1)
        if vgg_useBN: attention_x_7_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_7_2')(
            attention_x_7_2)
        attention_x_7_2 = Activation('relu', name='attention_x_7_2_relu')(attention_x_7_2)

        attention_x_8_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg),
                                 name='attention_x_8_1')(attention_x_7_2)
        if vgg_useBN: attention_x_8_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_8_1')(attention_x_8_1)
        attention_x_8_1 = Activation('relu', name='attention_x_8_1_relu')(attention_x_8_1)
        attention_x_8_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=l2(l2_reg), name='attention_x_8_2')(attention_x_8_1)
        if vgg_useBN: attention_x_8_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False,
                                                           name='bn_attention_x_8_2')(attention_x_8_2)
        attention_x_8_2 = Activation('relu', name='attention_x_8_2_relu')(attention_x_8_2)

        attention_x_3c_attn = Conv2D(n_boxes[0], (3, 3), padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=l2(l2_reg), name='attention_x_3c_attn')(
            attention_x_3c_feat)
        attention_x_4f_attn = Conv2D(n_boxes[1], (3, 3), padding='same', kernel_initializer='he_normal',
                                     kernel_regularizer=l2(l2_reg), name='attention_x_4f_attn')(
            attention_x_4f_feat)
        attention_x_attn = Conv2D(n_boxes[2], (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='attention_x_attn')(attention_x_feat)
        attention_x_6_attn = Conv2D(n_boxes[3], (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='attention_x_6_attn')(
            attention_x_6_2)
        attention_x_7_attn = Conv2D(n_boxes[4], (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='attention_x_7_attn')(
            attention_x_7_2)
        attention_x_8_attn = Conv2D(n_boxes[5], (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='attention_x_8_attn')(
            attention_x_8_2)

        attention_x_3c_attn_reshape = Reshape((-1,), name='attention_x_3c_attn_reshape')(
            attention_x_3c_attn)
        attention_x_4f_attn_reshape = Reshape((-1,), name='attention_x_4f_attn_reshape')(
            attention_x_4f_attn)
        attention_x_attn_reshape = Reshape((-1,), name='attention_x_attn_reshape')(attention_x_attn)
        attention_x_6_attn_reshape = Reshape((-1,), name='attention_x_6_attn_reshape')(attention_x_6_attn)
        attention_x_7_attn_reshape = Reshape((-1,), name='attention_x_7_attn_reshape')(attention_x_7_attn)
        attention_x_8_attn_reshape = Reshape((-1,), name='attention_x_8_attn_reshape')(attention_x_8_attn)

        attention_logits = Concatenate(axis=1, name='attention_attention')([attention_x_3c_attn_reshape,
                                                                            attention_x_4f_attn_reshape,
                                                                            attention_x_attn_reshape,
                                                                            attention_x_6_attn_reshape,
                                                                            attention_x_7_attn_reshape,
                                                                            attention_x_8_attn_reshape])

        l = int(flow_x.shape[1])
        h = int(flow_x.shape[2])
        w = int(flow_x.shape[3])

        flow_embedding = conv3d_bn(flow_x, temporal_channels[0], l, 1, 1, strides=(l, 1, 1),
                                   padding='valid', l2_reg=l2_reg,
                                   name='flow_temporal_embedding_1')
        flow_embedding = conv3d_bn(flow_embedding, temporal_channels[1], 1, 2, 2, strides=(1, 2, 2),
                                   padding='same', l2_reg=l2_reg,
                                   name='flow_temporal_embedding_2')

        temporal_embedding = Flatten(name='flow_to_vec')(flow_embedding)

        if whatToHelpWhere:
            i3d_toy_logits = Dense(n_classes, activation='linear', kernel_regularizer=l2(l2_reg),
                               name='i3d_toy_logits')(temporal_embedding)


    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(img_input)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(
            x1)

    conv1_1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv1_1')(x1)
    if vgg_useBN: conv1_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_1_1')(conv1_1)
    conv1_1 = Activation('relu', name='conv1_1_relu')(conv1_1)
    conv1_2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv1_2')(conv1_1)
    if vgg_useBN: conv1_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_1_2')(conv1_2)
    conv1_2 = Activation('relu', name='conv1_2_relu')(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv2_1')(pool1)
    if vgg_useBN: conv2_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_2_1')(conv2_1)
    conv2_1 = Activation('relu', name='conv2_1_relu')(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv2_2')(conv2_1)
    if vgg_useBN: conv2_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_2_2')(conv2_2)
    conv2_2 = Activation('relu', name='conv2_2_relu')(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv3_1')(pool2)
    if vgg_useBN: conv3_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_3_1')(conv3_1)
    conv3_1 = Activation('relu', name='conv3_1_relu')(conv3_1)
    conv3_2 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv3_2')(conv3_1)
    if vgg_useBN: conv3_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_3_2')(conv3_2)
    conv3_2 = Activation('relu', name='conv3_2_relu')(conv3_2)
    conv3_3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv3_3')(conv3_2)
    if vgg_useBN: conv3_3 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_3_3')(conv3_3)
    conv3_3 = Activation('relu', name='conv3_3_relu')(conv3_3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv4_1')(pool3)
    if vgg_useBN: conv4_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_4_1')(conv4_1)
    conv4_1 = Activation('relu', name='conv4_1_relu')(conv4_1)
    conv4_2 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv4_2')(conv4_1)
    if vgg_useBN: conv4_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_4_2')(conv4_2)
    conv4_2 = Activation('relu', name='conv4_2_relu')(conv4_2)
    conv4_3 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv4_3')(conv4_2)
    if vgg_useBN: conv4_3 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_4_3')(conv4_3)
    conv4_3 = Activation('relu', name='conv4_3_relu')(conv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv5_1')(pool4)
    if vgg_useBN: conv5_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_5_1')(conv5_1)
    conv5_1 = Activation('relu', name='conv5_1_relu')(conv5_1)
    conv5_2 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv5_2')(conv5_1)
    if vgg_useBN: conv5_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_5_2')(conv5_2)
    conv5_2 = Activation('relu', name='conv5_2_relu')(conv5_2)
    conv5_3 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv5_3')(conv5_2)
    if vgg_useBN: conv5_3 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_5_3')(conv5_3)
    conv5_3 = Activation('relu', name='conv5_3_relu')(conv5_3)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg), name='fc6')(pool5)
    if vgg_useBN: fc6 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_fc6')(fc6)
    fc6 = Activation('relu', name='fc6_relu')(fc6)

    fc7 = Conv2D(1024, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                 name='fc7')(fc6)
    if vgg_useBN: fc7 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_fc7')(fc7)
    fc7 = Activation('relu', name='fc7_relu')(fc7)

    conv6_1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv6_1')(fc7)
    if vgg_useBN: conv6_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_6_1')(conv6_1)
    conv6_1 = Activation('relu', name='conv6_1_relu')(conv6_1)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)
    if vgg_useBN: conv6_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_6_2')(conv6_2)
    conv6_2 = Activation('relu', name='conv6_2_relu')(conv6_2)

    conv7_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv7_1')(conv6_2)
    if vgg_useBN: conv7_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_7_1')(conv7_1)
    conv7_1 = Activation('relu', name='conv7_1_relu')(conv7_1)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)
    if vgg_useBN: conv7_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_7_2')(conv7_2)
    conv7_2 = Activation('relu', name='conv7_2_relu')(conv7_2)

    conv8_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv8_1')(conv7_2)
    if vgg_useBN: conv8_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_8_1')(conv8_1)
    conv8_1 = Activation('relu', name='conv8_1_relu')(conv8_1)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)
    if vgg_useBN: conv8_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_8_2')(conv8_2)
    conv8_2 = Activation('relu', name='conv8_2_relu')(conv8_2)

    conv9_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),
                     name='conv9_1')(conv8_2)
    if vgg_useBN: conv9_1 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_9_1')(conv9_1)
    conv9_1 = Activation('relu', name='conv9_1_relu')(conv9_1)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)
    if vgg_useBN: conv9_2 = BatchNormalization(momentum=vgg_BN_momentum, scale=False, name='bn_9_2')(conv9_2)
    conv9_2 = Activation('relu', name='conv9_2_relu')(conv9_2)

    conv4_3_norm = Lambda(identity_layer, name='conv4_3_identity')(conv4_3)

    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)

    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                             aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                             this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords,
                                             name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                    aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                    clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords,
                                    name='fc7_mbox_priorbox')(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                        aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                        this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                        aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                        this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                        aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                        this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                        aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                        this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)

    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(
        conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    def expand_dim_layer(tensor):
        return K.expand_dims(tensor, axis=-1)

    attn_conf_logits = Lambda(expand_dim_layer, name='expand_dim_attn')(attention_logits)

    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    if whereToHelpWhat:
        conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same',
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
        fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
        conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
        conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
        conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
        conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)
        conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(
            conv4_3_norm_mbox_conf)
        fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
        conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
        conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
        conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
        conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
        mbox_conf_logits = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                                  fc7_mbox_conf_reshape,
                                                                  conv6_2_mbox_conf_reshape,
                                                                  conv7_2_mbox_conf_reshape,
                                                                  conv8_2_mbox_conf_reshape,
                                                                  conv9_2_mbox_conf_reshape])

        mbox_conf_scores = Activation('softmax', name='mbox_conf_scores')(mbox_conf_logits)

    def myAvg(tensors):
        return (tensors[0] + tensors[1]) / 2.

    if whatHelpInside:
        assert (whatToHelpWhere and whereToHelpWhat), 'if whatHelpInside, then whatTohelpWhere and whereToHelpWhat must be true'

        classPredSimilarity = Lambda(cosineSimi, name='what_to_help_where_correlation')(
            [i3d_toy_logits, mbox_conf_logits])

        classPredSimilarity = Lambda(rescaleTensor1, name='reascale_simi')(classPredSimilarity)
        attn_conf_logits = Lambda(rescaleTensor1, name='rescale_attn')(
            attn_conf_logits)
        attn_conf_logits = Add(name='what_to_help_where_add')([classPredSimilarity, attn_conf_logits])

    if whereHelpInside:
        assert (whatToHelpWhere and whereToHelpWhat), 'if whereHelpInside, then whatTohelpWhere and whereToHelpWhat must be true'
        if softArgmax:
            attn_mbox_class_logits = Lambda(softPredBoxClassification, name='where_to_help_what_attn_box_classification')(
                [attn_conf_logits, mbox_conf_logits])
        else:
            attn_mbox_class_logits = Lambda(predBoxClassification, name='where_to_help_what_attn_box_classification')(
                [attn_conf_logits, mbox_conf_logits])
        attn_mbox_class_logits = Lambda(rescaleTensor2, name='rescale_mbox_class')(attn_mbox_class_logits)
        i3d_toy_logits = Lambda(rescaleTensor2, name='rescale_i3d_logits')(i3d_toy_logits)  # rescale to [-1, 1]
        i3d_toy_logits = Add(name='where_to_help_what_add')(
            [attn_mbox_class_logits, i3d_toy_logits])

    attn_conf_scores = Lambda(squeezeLayer, name='squeeze_back_attn_conf')(attn_conf_logits)
    attn_conf_scores = Activation('softmax', name='one_hot_attn_conf_scores')(attn_conf_scores)
    attn_conf_scores = Lambda(expand_dim_layer, name='attn_conf_scores_inflatted')(attn_conf_scores)
    zeros_attn = Lambda(my_zeros_like, name='zeros_attn')(attn_conf_logits)

    if whatToHelpWhere:
        i3d_toy_scores = Activation('softmax', name='i3d_toy_scores')(i3d_toy_logits)
        zeros_toy = Lambda(my_zeros_like, name='zeros_toy')(i3d_toy_logits)
    if whereToHelpWhat:
        predictions = Concatenate(axis=2, name='predictions')(
            [mbox_conf_scores, mbox_loc, mbox_priorbox, mbox_conf_logits, zeros_attn, zeros_attn])
    else:
        predictions = Concatenate(axis=2, name='predictions')(
            [mbox_loc, mbox_priorbox, zeros_attn, zeros_attn])

    attention_output = Concatenate(axis=-1, name='attention_output')([attn_conf_logits, attn_conf_scores])
    #attn_conf_scores = Lambda(squeezeLayer, name='attn_conf_scores')(attn_conf_scores)

    zeros_attn = Lambda(squeezeLayer, name='zeros_attn_squeeze')(zeros_attn)

    if mode == 'training' or mode == 'test':
        if whatToHelpWhere:
            if useRGBStream and useFlowStream:
                model = Model(inputs=[img_input, rgb_input, flow_input],
                              outputs=[predictions, zeros_attn, attention_output, zeros_toy, i3d_toy_scores])
            elif useRGBStream:
                model = Model(inputs=[img_input, rgb_input],
                              outputs=[predictions, zeros_attn, attention_output, zeros_toy, i3d_toy_scores])
            elif useFlowStream:
                model = Model(inputs=[img_input, flow_input],
                              outputs=[predictions, zeros_attn, attention_output, zeros_toy, i3d_toy_scores])
        else:
            if useRGBStream and useFlowStream:
                model = Model(inputs=[img_input, rgb_input, flow_input],
                              outputs=[predictions, zeros_attn, attention_output])
            elif useRGBStream:
                model = Model(inputs=[img_input, rgb_input], outputs=[predictions, zeros_attn, attention_output])
            elif useFlowStream:
                model = Model(inputs=[img_input, flow_input], outputs=[predictions, zeros_attn, attention_output])


    else:
        raise ValueError(
            "`mode` must be one of 'training' or 'test', but received '{}'.".format(mode))

    return model


class mrLoss:
    '''
    The mindReader loss
    '''

    def __init__(self,
                 num_class=24,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.,
                 whereToHelpWhat=True,
                 whatToHelpWhere=True):
        '''
        Arguments:
            neg_pos_ratio (int, optional): The maximum ratio of negative (i.e. background)
                to positive ground truth boxes to include in the loss computation.
                There are no actual background ground truth boxes of course, but `y_true`
                contains anchor boxes labeled with the background class. Since
                the number of background boxes in `y_true` will usually exceed
                the number of positive boxes by far, it is necessary to balance
                their influence on the loss. Defaults to 3 following the paper.
            n_neg_min (int, optional): The minimum number of negative ground truth boxes to
                enter the loss computation *per batch*. This argument can be used to make
                sure that the model learns from a minimum number of negatives in batches
                in which there are very few, or even none at all, positive ground truth
                boxes. It defaults to 0 and if used, it should be set to a value that
                stands in reasonable proportion to the batch size used for training.
            alpha (float, optional): A factor to weight the localization loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
            beta (float, optional): A factor to weight the classification loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
            gamma: A factor to weight the temporal classification loss and spatial detection loss.
            whereToHelpWhat: bool, whether to use the whereToHelpWhatMoudule
            whatToHelpWhere: bool, whether to use the whatToHelpWhereMoudule
        '''
        self.num_class = num_class
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.whereToHelpWhat = whereToHelpWhat
        self.whatToHelpWhere = whatToHelpWhere

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.
        References:
            https://arxiv.org/abs/1504.08083
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        Compute the softmax log loss.
        '''
        y_pred = tf.maximum(y_pred, 1e-15)
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_box_loss(self, y_true, y_pred):
        self.alpha = tf.constant(self.alpha)
        boxPred = y_pred
        boxPredGT = y_true
        if self.whereToHelpWhat:
            mbox_conf_scores, mbox_loc, attn_conf_scores = \
                boxPred[:, :, :self.num_class], boxPred[:, :, self.num_class:self.num_class + 4], boxPred[:, :, -1]
            mbox_conf_scoresGT, mbox_locGT, netralFlag, attn_conf_scoresGT = \
                tf.to_float(boxPredGT[:, :, :self.num_class]), boxPredGT[:, :, self.num_class:self.num_class + 4], tf.to_float(
                    boxPredGT[:, :, -2:]), tf.to_float(boxPredGT[:, :, -1])
        else:
            mbox_loc, attn_conf_scores = boxPred[:, :, :4], boxPred[:, :, 13]
            mbox_locGT, netralFlag, attn_conf_scoresGT = boxPredGT[:, :, :4], tf.to_float(
                boxPredGT[:, :, 12:]), tf.to_float(boxPredGT[:, :, 13])

        batch_size = tf.shape(boxPred)[0]
        n_boxes = tf.shape(boxPred)[1]

        localization_loss = tf.to_float(
            self.smooth_L1_loss(mbox_locGT, mbox_loc))

        negatives = tf.ones([batch_size, n_boxes]) - tf.to_float(K.any(netralFlag, axis=-1))
        positives = tf.to_float(K.all(netralFlag, axis=-1))
        n_positive = tf.reduce_sum(positives)

        if self.whereToHelpWhat:
            classification_loss = K.categorical_crossentropy(mbox_conf_scoresGT,
                                                             mbox_conf_scores)
            pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)
        if self.whereToHelpWhat:
            total_loss = (loc_loss + self.alpha * pos_class_loss) / tf.maximum(1.0,
                                                                               n_positive)  # In case `n_positive == 0`
        else:
            total_loss = loc_loss / tf.maximum(1.0, n_positive)
        total_loss = total_loss * tf.to_float(batch_size)

        return total_loss

    def zero_loss(self, y_true, y_pred):
        '''
        Simply return 0 for the 2nd pair of output of global class logits
        Note that when there are multiple outputs, keras returns it in a zip way
        '''
        batch_size = tf.shape(y_pred)[0]
        return tf.zeros(batch_size)

    def global_class_loss(self, y_true, y_pred):
        '''
        Compute the loss for the 3rd pair of output of gobal class scores
        Note that when there are multiple outputs, keras returns it in a zip way
        '''
        return K.categorical_crossentropy(y_true, y_pred)

    def global_attn_loss_no_mining(self, y_true, y_pred):
        '''
        Compute the loss for the attention score without hard negative mining
        '''
        pred_attn = y_pred[:, :, 1]
        true_attn = y_true[:, :, 1]
        return K.binary_crossentropy(true_attn, pred_attn)

    def global_attn_loss_mining(self, y_true, y_pred):
        '''
        Compute the loss for the attention score with hard negative mining
        '''

        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)

        netralFlag, attn_conf_scoresGT = y_true, y_true[:, :, 1]
        attn_conf_scores = y_pred[:, :, 1]

        batch_size = tf.shape(netralFlag)[0]
        n_boxes = tf.shape(netralFlag)[1]

        negatives = tf.ones([batch_size, n_boxes]) - tf.to_float(
            K.any(netralFlag, axis=-1))
        positives = tf.to_float(K.all(netralFlag, axis=-1))
        n_positive = tf.reduce_sum(positives)
        attn_loss_all = K.binary_crossentropy(attn_conf_scoresGT, attn_conf_scores)

        pos_attn_loss = tf.reduce_sum(attn_loss_all * positives, axis=-1)
        neg_attn_loss_all = attn_loss_all * negatives
        n_neg_losses = tf.count_nonzero(neg_attn_loss_all,
                                        dtype=tf.int32)

        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min),
                                     n_neg_losses)

        def f1():
            return tf.zeros([batch_size])

        def f2():

            neg_attn_loss_all_1D = tf.reshape(neg_attn_loss_all, [-1])
            values, indices = tf.nn.top_k(neg_attn_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False)
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(
                                               neg_attn_loss_all_1D))
            negatives_keep = tf.to_float(
                tf.reshape(negatives_keep, [batch_size, n_boxes]))
            neg_attn_loss = tf.reduce_sum(attn_loss_all * negatives_keep, axis=-1)
            return neg_attn_loss

        neg_attn_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        attn_loss = pos_attn_loss + neg_attn_loss

        attn_loss = attn_loss * tf.to_float(batch_size) / tf.maximum(1.0, n_positive)

        return attn_loss


class SSDInputEncoder:

    def __init__(self,
                 img_height=300,
                 img_width=300,
                 n_classes=24,
                 predictor_sizes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                 min_scale=None,
                 max_scale=None,
                 scales=[0.07, 0.15, 0.32, 0.49, 0.66, 0.83, 1.0],
                 aspect_ratios_global=None,
                 aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                          [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                          [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                          [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                          [1.0, 2.0, 0.5],
                                          [1.0, 2.0, 0.5]],
                 two_boxes_for_ar1=True,
                 steps=[8, 16, 32, 64, 100, 300],
                 offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                 clip_boxes=False,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 matching_type='multi',
                 pos_iou_threshold=0.5,
                 neg_iou_limit=0.3,
                 border_pixels='half',
                 coords='centroids',
                 normalize_coords=True,
                 attn_id=[-2, -1]):

        predictor_sizes = np.array(predictor_sizes)
        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if (len(scales) != predictor_sizes.shape[0] + 1):
                raise ValueError(
                    "It must be either scales is None or len(scales) == len(predictor_sizes)+1, but len(scales) == {} and len(predictor_sizes)+1 == {}".format(
                        len(scales), len(predictor_sizes) + 1))
            scales = np.array(scales)
            if np.any(scales <= 0):
                raise ValueError(
                    "All values in `scales` must be greater than 0, but the passed list of scales is {}".format(scales))
        else:
            if not 0 < min_scale <= max_scale:
                raise ValueError(
                    "It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(
                        min_scale, max_scale))

        if not (aspect_ratios_per_layer is None):
            if (len(aspect_ratios_per_layer) != predictor_sizes.shape[0]):
                raise ValueError(
                    "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == len(predictor_sizes), but len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(
                        len(aspect_ratios_per_layer), len(predictor_sizes)))
            for aspect_ratios in aspect_ratios_per_layer:
                if np.any(np.array(aspect_ratios) <= 0):
                    raise ValueError("All aspect ratios must be greater than zero.")
        else:
            if (aspect_ratios_global is None):
                raise ValueError(
                    "At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` must not be `None`.")
            if np.any(np.array(aspect_ratios_global) <= 0):
                raise ValueError("All aspect ratios must be greater than zero.")

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        if not (coords == 'minmax' or coords == 'centroids' or coords == 'corners'):
            raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

        if (not (steps is None)) and (len(steps) != predictor_sizes.shape[0]):
            raise ValueError("You must provide at least one step value per predictor layer.")

        if (not (offsets is None)) and (len(offsets) != predictor_sizes.shape[0]):
            raise ValueError("You must provide at least one offset value per predictor layer.")


        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        if (scales is None):
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes) + 1)
        else:
            self.scales = scales
        if (aspect_ratios_per_layer is None):
            self.aspect_ratios = [aspect_ratios_global] * predictor_sizes.shape[0]
        else:
            self.aspect_ratios = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        if not (steps is None):
            self.steps = steps
        else:
            self.steps = [None] * predictor_sizes.shape[0]
        if not (offsets is None):
            self.offsets = offsets
        else:
            self.offsets = [None] * predictor_sizes.shape[0]
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.matching_type = matching_type
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
        self.coords = coords
        self.normalize_coords = normalize_coords
        self.attn_id = attn_id
        if not (aspect_ratios_per_layer is None):
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if (1 in aspect_ratios) & two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios_global) + 1
            else:
                self.n_boxes = len(aspect_ratios_global)

        self.boxes_list = []

        self.wh_list_diag = []  # Box widths and heights for each predictor layer
        self.steps_diag = []  # Horizontal and vertical distances between any two boxes for each predictor layer
        self.offsets_diag = []  # Offsets for each predictor layer
        self.centers_diag = []  # Anchor box center points as `(cy, cx)` for each predictor layer

        for i in range(len(self.predictor_sizes)):
            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(
                feature_map_size=self.predictor_sizes[i],
                aspect_ratios=self.aspect_ratios[i],
                this_scale=self.scales[i],
                next_scale=self.scales[i + 1],
                this_steps=self.steps[i],
                this_offsets=self.offsets[i],
                diagnostics=True)
            self.boxes_list.append(boxes)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)

        count_total_boxes = 0
        for i, n_box in enumerate(self.n_boxes):
            count_total_boxes += n_box * predictor_sizes[i][0] * predictor_sizes[i][1]

        self.total_boxes = count_total_boxes

        print(self.total_boxes)

    def __call__(self, ground_truth_labels, return_mode='general', diagnostics=False):

        class_id = 0
        cx = 1
        cy = 2
        w = 3
        h = 4

        batch_size = len(ground_truth_labels)


        y_encoded = self.generate_encoding_template(batch_size=batch_size, diagnostics=False)
        y_encoded[:, :, self.attn_id] = 0.
        n_boxes = y_encoded.shape[1]
        class_vectors = np.eye(self.n_classes)

        for i in range(batch_size):

            if ground_truth_labels[
                i].size == 0: continue
            labels = ground_truth_labels[i].astype(np.float)

            classes_one_hot = class_vectors[labels[:, class_id].astype(
                np.int)]

            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [cx, cy, w, h]]],
                                            axis=-1)

            similarities = iou(labels[:, [cx, cy, w, h]], y_encoded[i, :, self.n_classes:self.n_classes + 4], coords=self.coords,
                               mode='outer_product', border_pixels=self.border_pixels)
            bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)

            if return_mode == 'onehot':
                similarities[0, similarities[0, :] < self.pos_iou_threshold] = 0.
                one_hot_attn_vectors = np.eye(self.total_boxes)
                return one_hot_attn_vectors[bipartite_matches], similarities[0, :], vec_softmax(similarities[0, :])
            y_encoded[i, bipartite_matches, :self.n_classes] = classes_one_hot
            y_encoded[i, bipartite_matches, self.n_classes:self.n_classes + 4] = labels[:, [cx, cy, w, h]]
            y_encoded[i, bipartite_matches, -2 - self.n_classes:-2] = classes_one_hot
            y_encoded[i, bipartite_matches, -2] = y_encoded[i, bipartite_matches, -1] = 1.

            similarities[:, bipartite_matches] = 0.

            if self.matching_type == 'multi':
                matches = match_multi(weight_matrix=similarities, threshold=self.pos_iou_threshold)

                y_encoded[i, matches[1], :self.n_classes] = classes_one_hot
                y_encoded[i, matches[1], self.n_classes:self.n_classes + 4] = labels[:, [cx, cy, w, h]]
                y_encoded[i, matches[1], -2 - self.n_classes:-2] = classes_one_hot
                y_encoded[i, matches[1], -2] = y_encoded[i, [matches[1]], -1] = 1.
                similarities[:, matches[1]] = 0.

            neutral_boxes = np.where(similarities >= self.neg_iou_limit)

            y_encoded[neutral_boxes[0], neutral_boxes[
                1], -2] = 1.
            y_encoded[neutral_boxes[0], neutral_boxes[
                1], -1] = 0.


        if self.coords == 'centroids':
            y_encoded[:, :, [self.n_classes, self.n_classes + 1]] -= y_encoded[:, :, [self.n_classes + 4, self.n_classes + 5]]  # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:, :, [self.n_classes, self.n_classes + 1]] /= y_encoded[:, :, [self.n_classes + 6, self.n_classes + 7]] * y_encoded[:, :, [self.n_classes + 8,
                                                                                                                                                  self.n_classes + 9]]  # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
            y_encoded[:, :, [self.n_classes + 2, self.n_classes + 3]] /= y_encoded[:, :, [self.n_classes + 6, self.n_classes + 7]]  # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:, :, [self.n_classes + 2, self.n_classes + 3]] = np.log(y_encoded[:, :, [self.n_classes + 2, self.n_classes + 3]]) / y_encoded[:, :, [self.n_classes + 10,
                                                                                                                                                             self.n_classes + 11]]  # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)


        if diagnostics:
            y_matched_anchors = np.copy(y_encoded)
            y_matched_anchors[:, :, -12:-8] = 0
            return y_encoded, y_matched_anchors
        else:
            return y_encoded

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale,
                                        this_steps=None,
                                        this_offsets=None,
                                        diagnostics=False):

        size = min(self.img_height, self.img_width)
        wh_list = []
        for ar in aspect_ratios:
            if (ar == 1):
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        if (this_steps is None):
            step_height = self.img_height / feature_map_size[0]
            step_width = self.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height,
                         feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width,
                         feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]

        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
        if self.coords == 'centroids':
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids',
                                               border_pixels='half')
        elif self.coords == 'minmax':
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax',
                                               border_pixels='half')

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor

    def generate_encoding_template(self, batch_size, diagnostics=False):

        boxes_batch = []
        for boxes in self.boxes_list:
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances
        attn_tensor = np.zeros((batch_size, boxes_tensor.shape[1], 1))

        y_encoding_template = np.concatenate(
            (classes_tensor, boxes_tensor, boxes_tensor, variances_tensor, classes_tensor, attn_tensor, attn_tensor),
            axis=2)

        if diagnostics:
            return y_encoding_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encoding_template


def encodeLabelsEpic():

    baseFolder = './'

    labelEncoder = SSDInputEncoder(n_classes=53)

    trainList = open('{}epic_train.txt'.format(baseFolder), 'r')
    trainSamples = trainList.readlines()
    trainList.close()
    testList = open('{}epic_test.txt'.format(baseFolder), 'r')
    testSamples = testList.readlines()
    testList.close()


    print(len(trainSamples))
    print(len(testSamples))

    trainLabelsF = h5py.File('{}epic_train_labels.h5'.format(baseFolder), 'a')
    trainLabels = trainLabelsF.create_dataset('data', shape=(len(trainSamples), 8732, 120))

    trainOneHotAttnF = h5py.File('{}epic_train_onehot.h5'.format(baseFolder), 'a')
    trainOneHotAttn = trainOneHotAttnF.create_dataset('data', shape=(len(trainSamples), 3, 8732))

    testLabelsF = h5py.File('{}epic_test_labels.h5'.format(baseFolder), 'a')
    testLabels = testLabelsF.create_dataset('data', shape=(len(testSamples), 8732, 120))

    testOneHotAttnF = h5py.File('{}epic_test_onehot.h5'.format(baseFolder), 'a')
    testOneHotAttn = testOneHotAttnF.create_dataset('data', shape=(len(testSamples), 3, 8732))

    for k, sample in enumerate(trainSamples):
        print('Train: {}/{}'.format(k + 1, len(trainSamples)))
        entries = sample.strip().split(' ')

        gt = np.array([[[int(entries[-1]), float(entries[-5]), float(entries[-4]), float(entries[-3]),
                         float(entries[-2])]]])
        trainLabels[k, :, :] = labelEncoder(gt)[0]
        trainOneHotAttn[k, 0, :], trainOneHotAttn[k, 1, :], trainOneHotAttn[k, 2, :] = labelEncoder(gt, return_mode='onehot')

    trainLabelsF.close()
    trainOneHotAttnF.close()

    for k, sample in enumerate(testSamples):
        print('Test: {}/{}'.format(k + 1, len(testSamples)))
        entries = sample.strip().split(' ')

        gt = np.array([[[int(entries[-1]), float(entries[-5]), float(entries[-4]), float(entries[-3]),
                         float(entries[-2])]]])
        testLabels[k, :, :] = labelEncoder(gt)[0]
        testOneHotAttn[k, 0, :], testOneHotAttn[k, 1, :], testOneHotAttn[k, 2, :] = labelEncoder(gt, return_mode='onehot')
    testLabelsF.close()
    testOneHotAttnF.close()



def vec_softmax(vec):
    e = np.exp(vec - np.amax(vec))
    return e / np.sum(e)


def myGenerator(args, split='train', shuffle=True):

    print('Epic generator!')
    baseFolder = './'
    classNum = 53

    sampleInfoF = open('{}epic_{}.txt'.format(baseFolder, split))
    sampleInfo = sampleInfoF.readlines()
    sampleInfoF.close()

    classGT = np.empty(len(sampleInfo))

    for i, line in enumerate(sampleInfo):
        entries = line.strip().split(' ')
        classGT[i] = int(entries[-1])

    flowData = h5py.File('{}epic_{}_flows_15.h5'.format(baseFolder, split), 'r')['data']
    rgbData = h5py.File('{}epic_{}_imgs_15.h5'.format(baseFolder, split), 'r')['data']
    frameData = h5py.File('{}epic_{}_imgs_15.h5'.format(baseFolder, split), 'r')['data']
    labelData = h5py.File('{}epic_{}_labels.h5'.format(baseFolder, split), 'r')['data']

    oneHotLabelData = h5py.File('{}epic_{}_onehot.h5'.format(baseFolder, split), 'r')['data']

    onehot_encoding = np.eye(classNum)

    numSamples = labelData.shape[0]

    assert numSamples == len(classGT)

    print(numSamples)

    idxes = np.arange(numSamples)
    count = 0

    n = int((args.sequence_length - 1) / 2)

    batch_size = args.batch_size

    frameBatch = np.empty((batch_size, 300, 300, 3))
    rgbBatch = np.empty((batch_size, args.sequence_length, args.input_height, args.input_width, 3))
    flowBatch = np.empty((batch_size, args.sequence_length, args.input_height, args.input_width, 2))
    globalClassBatch = np.empty((batch_size, classNum))
    globalAttentionBatch = np.empty((batch_size, 8732, 2))
    boxBatch = np.empty((batch_size, 8732, 120))

    while True:
        if shuffle:
            np.random.shuffle(idxes)
        for idx in idxes:
            absIdx = idx * 16 + 8
            frameBatch[count, :, :, :] = (frameData[absIdx, :, :, :] + 1.) * 127.5  # scale [-1, 1] back to [0, 255]
            rgbBatch[count, :, :, :, :] = rgbData[absIdx - n: absIdx + n + 1, :, :, :]
            flowBatch[count, :, :, :, :] = flowData[absIdx - n: absIdx + n + 1, :, :, :]
            globalClassBatch[count, :] = onehot_encoding[int(classGT[idx]), :]
            boxBatch[count, :, :] = labelData[idx, :, :]
            globalAttentionBatch[count, :, 0] = globalAttentionBatch[count, :, 1] = oneHotLabelData[idx, args.oneHotMode, :]
            count += 1
            if count == batch_size:
                count = 0
                yield [frameBatch, rgbBatch, flowBatch], [boxBatch, globalAttentionBatch[:, :, 1],
                                                          globalAttentionBatch, globalClassBatch,
                                                          globalClassBatch]


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


def train(args):
    ##################### Below are fixed parameters for the SSD backbone

    img_height = 300  # Height of the model input images
    img_width = 300  # Width of the model input images
    img_channels = 3  # Number of color channels of the model input images
    mean_color = [123, 117,
                  104]  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
    swap_channels = [2, 1,
                     0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
    n_classes = 53  # Number of positive classes
    scales_toy = [0.07, 0.15, 0.32, 0.49, 0.66, 0.83, 1.0]
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 100,
             300]  # The space between two adjacent anchor box center points for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
               0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2,
                 0.2]  # The variances by which the encoded target coordinates are divided as in the original implementation
    normalize_coords = True
    coords = 'centroids'
    n_neg_min = 0

    ##############

    baseFolder = './'

    trainGenerator = myGenerator(args, 'train', shuffle=True)
    validGenerator = myGenerator(args, 'test', shuffle=False)

    myLoss = mrLoss(num_class=n_classes,
                    neg_pos_ratio=3,
                    n_neg_min=n_neg_min,
                    alpha=1.0,
                    beta=1.0,
                    gamma=1.0,
                    whereToHelpWhat=True,
                    whatToHelpWhere=True)

    myOptimizer = SGD(lr=args.init_learning_rate,
                      momentum=0.9,
                      decay=args.decay)



    if args.where_help_inside:
        assert args.softArgmax, 'Traditional argmax cannot have gradients propagate back. Use soft argmax by indicating --softArgmax insted'


    if args.num_gpu > 1:  # multiGPU training
        with tf.device('/cpu:0'):
        #with tf.device('/gpu:0'):
            oriModel = mindReader((img_height, img_width, img_channels),
                                    (args.sequence_length, args.input_height, args.input_width),
                                    n_classes=n_classes,
                                    mode='training',
                                    l2_regularization=args.l2_reg,
                                    min_scale=None,
                                    max_scale=None,
                                    scales=scales_toy,
                                    aspect_ratios_global=None,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    coords=coords,
                                    normalize_coords=normalize_coords,
                                    subtract_mean=mean_color,
                                    divide_by_stddev=None,
                                    swap_channels=swap_channels,
                                    vgg_useBN=True,
                                    vgg_BN_momentum=args.vgg_BN_momentum,
                                    i3d_useBN=True,
                                    i3d_BN_momentum=args.i3d_BN_momentum,
                                    whereToHelpWhat=True,
                                    whatToHelpWhere=True,
                                    whereHelpInside=args.where_help_inside,
                                    whatHelpInside=args.what_help_inside,
                                    useRGBStream=True,
                                    useFlowStream=True,
                                    temporal_channels=[256, 256],
                                    softArgmax=True
                                    )
        with tf.device('/cpu:0'):
            if args.load_exist_model:
                oriModel.load_weights(args.load_from, by_name=True)
            else:

                if os.path.exists('{}pretrained_spatial.h5'.format(baseFolder)):
                    oriModel.load_weights(
                        '{}pretrained_spatial.h5'.format(baseFolder), by_name=True)
                else:
                    oriModel.load_weights(
                        '{}VGG.h5'.format(baseFolder), by_name=True)

                oriModel.load_weights(
                    '{}rgb_stream.h5'.format(baseFolder), by_name=True)

                oriModel.load_weights(
                    '{}flow_stream.h5'.format(baseFolder), by_name=True)

        model = multi_gpu_model(oriModel, gpus=args.num_gpu)
        K.get_session().run(tf.global_variables_initializer())

    else:
        model = mindReader((img_height, img_width, img_channels),
                            (args.sequence_length, args.input_height, args.input_width),
                            n_classes=n_classes,
                            mode='training',
                            l2_regularization=args.l2_reg,
                            min_scale=None,
                            max_scale=None,
                            scales=scales_toy,
                            aspect_ratios_global=None,
                            aspect_ratios_per_layer=aspect_ratios,
                            two_boxes_for_ar1=two_boxes_for_ar1,
                            steps=steps,
                            offsets=offsets,
                            clip_boxes=clip_boxes,
                            variances=variances,
                            coords=coords,
                            normalize_coords=normalize_coords,
                            subtract_mean=mean_color,
                            divide_by_stddev=None,
                            swap_channels=swap_channels,
                            vgg_useBN=True,
                            vgg_BN_momentum=args.vgg_BN_momentum,
                            i3d_useBN=True,
                            i3d_BN_momentum=args.i3d_BN_momentum,
                            whereToHelpWhat=True,
                            whatToHelpWhere=True,
                            whereHelpInside=args.where_help_inside,
                            whatHelpInside=args.what_help_inside,
                            useRGBStream=True,
                            useFlowStream=True,
                            temporal_channels=[256, 256],
                            softArgmax=True
                            )
        if args.load_exist_model:
            model.load_weights(args.load_from, by_name=True)
        else:

            model.load_weights(
                '{}1_0.8_0.05_4_pretrained_ssd.h5'.format(baseFolder),
                by_name=True)

            model.load_weights('{}rgb_stream.h5'.format(baseFolder),
                               by_name=True)
            model.load_weights('{}flow_stream.h5'.format(baseFolder),
                               by_name=True)
    myMetrics = {}
    loss1 = myLoss.compute_box_loss
    loss3 = myLoss.global_one_hot_attn_loss

    loss2 = myLoss.zero_loss  # This non-sense part is due to some experiments we performed. We leave the cleaning work for the future.
    lossWeight2 = 0.0

    loss5 = myLoss.global_class_loss
    myMetrics['i3d_toy_scores'] = 'acc'
    loss4 = myLoss.zero_loss
    lossWeight4 = 0.0

    model.compile(
        loss=[loss1, loss2, loss3, loss4, loss5],
        loss_weights=[1.0, lossWeight2, 1.0, lossWeight4, 1.0], optimizer=myOptimizer,
        metrics=myMetrics)

    terminate_on_nan = TerminateOnNaN()


    weightsPath = '{}epic_mindreader.h5'

    myMonitor = 'val_loss'

    learning_rate_scheduler = ReduceLROnPlateau(monitor=myMonitor, factor=0.1, patience=args.patience, verbose=1)

    if args.num_gpu > 1:
        checkingPoint = MultiGPUCheckpointCallback(weightsPath, oriModel, verbose=1, monitor=myMonitor,
                                                   save_best_only=True, save_weights_only=False, period=1)
    else:
        checkingPoint = ModelCheckpoint(weightsPath, verbose=1, monitor=myMonitor, save_best_only=True,
                                        save_weights_only=False, period=1)
    lossHis = LossHistory()

    callbacks = [learning_rate_scheduler, checkingPoint, terminate_on_nan, lossHis]

    val_dataset_size = args.num_val_samples

    history = model.fit_generator(generator=trainGenerator,
                                  steps_per_epoch=args.steps_per_epoch,  # changable
                                  epochs=args.epochs,  # changable
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=validGenerator,
                                  #validation_steps=int(np.ceil(float(val_dataset_size) / args.batch_size)),
                                  validation_steps=val_dataset_size,
                                  initial_epoch=args.initial_epoch)


def test(args):
    ##################### Below are fixed parameters for the SSD backbone

    img_height = 300  # Height of the model input images
    img_width = 300  # Width of the model input images
    img_channels = 3  # Number of color channels of the model input images
    mean_color = [123, 117,
                  104]  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
    swap_channels = [2, 1,
                     0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
    n_classes = 53  # Number of positive classes
    scales_toy = [0.07, 0.15, 0.32, 0.49, 0.66, 0.83, 1.0]
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    steps = [8, 16, 32, 64, 100,
             300]  # The space between two adjacent anchor box center points for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
               0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2,
                 0.2]  # The variances by which the encoded target coordinates are divided as in the original implementation
    normalize_coords = True
    coords = 'centroids'

    ############################

    baseFolder = './'

    validGenerator = myGenerator(args, 'test', shuffle=False)

    model = mindReader((img_height, img_width, img_channels),
                       (args.sequence_length, args.input_height, args.input_width),
                       n_classes=n_classes,
                       mode='test',
                       l2_regularization=args.l2_reg,
                       min_scale=None,
                       max_scale=None,
                       scales=scales_toy,
                       aspect_ratios_global=None,
                       aspect_ratios_per_layer=aspect_ratios,
                       two_boxes_for_ar1=two_boxes_for_ar1,
                       steps=steps,
                       offsets=offsets,
                       clip_boxes=clip_boxes,
                       variances=variances,
                       coords=coords,
                       normalize_coords=normalize_coords,
                       subtract_mean=mean_color,
                       divide_by_stddev=None,
                       swap_channels=swap_channels,
                       vgg_useBN=True,
                       vgg_BN_momentum=args.vgg_BN_momentum,
                       i3d_useBN=True,
                       i3d_BN_momentum=args.i3d_BN_momentum,
                       whereToHelpWhat=True,
                       whatToHelpWhere=True,
                       whereHelpInside=args.where_help_inside,
                       whatHelpInside=args.what_help_inside,
                       useRGBStream=True,
                       useFlowStream=True,
                       temporal_channels=[256, 256],
                       softArgmax=False
                       )

    plot_model(model, './debug_model.png', show_shapes=True)

    if args.load_exist_model:
        model.load_weights(args.load_from, by_name=True)


    labelData = h5py.File('{}epic_test_labels.h5'.format(baseFolder), 'r')['data']

    samplesFile = open('{}epic_test.txt'.format(baseFolder), 'r')
    samples = samplesFile.readlines()
    samplesFile.close()

    val_dataset_size = len(samples)


    allPred = np.empty((val_dataset_size, 5))
    allGT = np.empty((val_dataset_size, 5))
    accumIOU = 0.
    accumCorrectPred = 0.
    accumCorrectAnchor = 0.
    accumCorrectBoxClass = 0.

    threshHolds = np.arange(0.5, 1.0, 0.05)
    Accs = np.zeros(10)

    for i, sample in enumerate(samples):
        print('ID: {}'.format(i))

        inputs, labels = next(validGenerator)
        entries = sample.strip().split(' ')


        predictions = model.predict(inputs)

        global_class_scores = predictions[4]

        boxPred = predictions[0]

        box_attn_scores = predictions[2][0, :, 1]

        box_class_scores = boxPred[0, :, :53]
        box_loc = boxPred[0, :, 53:57]
        anchor_loc = boxPred[0, :, 57:61]

        box_idx = np.argmax(box_attn_scores, axis=-1)
        print('Pred box index: {}'.format(box_idx))

        if labelData[i, box_idx, -2] == 1 and labelData[i, box_idx, -1] == 1:
            print('Correct Anchor!')
            accumCorrectAnchor += 1
        anchorAcc = accumCorrectAnchor / (i + 1)

        print('Anchor acc: {}'.format(anchorAcc))

        pred_loc = box_loc[box_idx, :]
        anchors = anchor_loc[box_idx, :]
        pred_box_class = box_class_scores[box_idx, :]

        pred_class = np.argmax(global_class_scores[0], axis=-1)

        box_pred_class = np.argmax(pred_box_class, axis=-1)

        xywh = np.empty((1, 4))

        xywh[0, 0] = pred_loc[0] * variances[0] * anchors[2] + anchors[0]
        xywh[0, 1] = pred_loc[1] * variances[1] * anchors[3] + anchors[1]
        xywh[0, 2] = np.exp(pred_loc[2] * variances[2]) * anchors[2]
        xywh[0, 3] = np.exp(pred_loc[3] * variances[3]) * anchors[3]

        allPred[i, 4] = pred_class
        allPred[i, 0] = xywh[0, 0]
        allPred[i, 1] = xywh[0, 1]
        allPred[i, 2] = xywh[0, 2]
        allPred[i, 3] = xywh[0, 3]

        gt_box = np.empty((1, 4))
        gt_box[0, 0] = float(entries[-5])
        gt_box[0, 1] = float(entries[-4])
        gt_box[0, 2] = float(entries[-3])
        gt_box[0, 3] = float(entries[-2])

        allGT[i, :4] = gt_box
        allGT[i, 4] = int(entries[-1])

        cIOU = iou(xywh, gt_box, coords='centroids', mode='element-wise')

        print('Current IOU: {}'.format(cIOU[0]))

        accumIOU += cIOU[0]
        mIOU = accumIOU / (i + 1)
        print('mIOU: {}'.format(mIOU))

        if pred_class == int(entries[-1]):
            accumCorrectPred += 1

            Accs[np.where(threshHolds < cIOU[0])] += 1.

        Acc_5 = Accs[0] / (i + 1)
        Acc_75 = Accs[5] / (i + 1)
        mAcc = np.sum(Accs / (i + 1)) / 10

        print('Acc_0.5: {}'.format(Acc_5))
        print('Acc_0.75: {}'.format(Acc_75))
        print('mAcc: {}'.format(mAcc))

        classAcc = accumCorrectPred / (i + 1)

        print('Classification acc: {}'.format(classAcc))

        if box_pred_class == int(entries[-1]):
            accumCorrectBoxClass += 1
        boxClassAcc = accumCorrectBoxClass / (i + 1)
        print('Box class acc: {}'.format(boxClassAcc))

    save_h5_data('/l/vision/v7/zehzhang/mindreaderv2/main/predict_epic.h5', 'data', allPred)


def epic_preprocess():
    import ast

    baseFolder = './'

    trainObjF = open('{}EPIC_train_object_labels.csv'.format(baseFolder), 'r')
    trainObj = trainObjF.readlines()[1:]
    trainObjF.close()

    objActF = open('{}EPIC_train_object_action_correspondence.csv'.format(baseFolder), 'r')
    objAct = objActF.readlines()[1:]
    objActF.close()

    trainActF = open('{}EPIC_train_action_labels.csv'.format(baseFolder), 'r')
    trainAct = trainActF.readlines()[1:]
    trainActF.close()

    obj2act = {}

    for line in objAct:
        entries = line.strip().split(',')
        thisKey = (entries[2], entries[-1], int(entries[1]))
        obj2act[thisKey] = int(entries[0])

    classCollection = {}
    class2noun = {}

    weirdResDict = {'P12_01': (720., 1280.), 'P12_02': (720., 1280.), 'P12_03': (720., 1280.), 'P12_04': (720., 1280.),
                    'P12_05': (1440., 1920.), 'P12_06': (1440., 1920.)}

    defaultRes = (1080., 1920.)

    classAppear = {}

    for line in trainAct:
        classStart = line.find('],[')
        classesStart = line.find('","')
        if classStart == -1 and classesStart == -1: continue
        if not classStart == -1:
            classInSeq = ast.literal_eval(line[classStart + 2:].strip())
            entries = line[:classStart].strip().split(',')
        else:  # not classesStart == -1
            classInSeq = ast.literal_eval(line[classesStart + 2:].strip()[1:-1])
            entries = line[:classesStart].strip().split(',')
        vid = entries[2]
        pid = entries[1]
        startFrame = int(entries[6])
        endFrame = int(entries[7])
        thisKey = (pid, vid)
        #print(thisKey)
        for activeClass in classInSeq:
            if activeClass not in classAppear.keys():
                classAppear[activeClass] = {}
            classAppear[activeClass][thisKey] = classAppear[activeClass].get(thisKey, []) + range(startFrame, endFrame + 1)
    cls2idx = {}
    count = 0
    for clsKey in classAppear.keys():
        cls2idx[clsKey] = count
        count += 1

    print(count)

    for i in range(count):
        classCollection[i] = []

    samplesPerClass = [0] * count

    for line in trainObj:
        if line.find('"[') == -1: continue  # check if there is a box
        boxStart = line.index('"[')
        entries = line[:boxStart - 1].strip().split(',')
        classId = int(entries[0])
        if not classId in cls2idx.keys(): continue # no such obj of this class in any active frames
        pid = entries[2]
        vid = entries[3]
        #frameId = int(entries[4])
        frameId = obj2act[(pid, vid, int(entries[4]))]
        box = ast.literal_eval(line[boxStart:].strip()[1:-1])
        if len(box) > 1: continue  # more than 1 box
        if (pid, vid) not in classAppear[classId].keys(): continue # exclude post-active and pro-active
        if frameId not in classAppear[classId][(pid, vid)]: continue
        if frameId - 20 < 1 or not os.path.isfile('{}EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train/{}/{}/'.format(baseFolder, pid, vid) + ('frame_%10d.jpg' % (frameId + 20)).replace(' ', '0')):
            continue # not enough frames ahead or behind
        if vid in weirdResDict.keys():
            height, width = weirdResDict[vid]
        else:
            height, width = defaultRes

        y, x, h, w = box[0]
        center_y = (y + h / 2.) / height
        center_x = (x + w / 2.) / width
        normed_h = h / height
        normed_w = w / width

        samplesPerClass[cls2idx[classId]] += 1
        class2noun[cls2idx[classId]] = entries[1]
        classCollection[cls2idx[classId]].append([pid, vid, frameId, center_x, center_y, normed_w, normed_h, cls2idx[classId]])

    T = 1000

    deleted = 0

    trainF = open('{}epic_train.txt'.format(baseFolder), 'w')
    testF = open('{}epic_test.txt'.format(baseFolder), 'w')

    for key in classCollection.keys():
        if len(classCollection[key]) < T:
            deleted += 1
            continue
        for sample in classCollection[key]:
            if np.random.rand() > 0.1: # 90% for training
                trainF.write('{} {} {} {} {} {} {} {}\n'.format(sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7] - deleted))
            else:
                testF.write('{} {} {} {} {} {} {} {}\n'.format(sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7] - deleted))

    trainF.close()
    testF.close()

    samPerClsF = open('{}epicSamplesPerClass.txt'.format(baseFolder), 'w')
    for i, ent in enumerate(samplesPerClass):
        if ent >= 1000:
            samPerClsF.write('{} {}\n'.format(class2noun[i] if i in class2noun.keys() else 'Unknown', ent))
    samPerClsF.close()

    samplesPerClass = np.array(samplesPerClass)
    print(np.sum(samplesPerClass >= T))
    print(np.sum((samplesPerClass >= T) * samplesPerClass))
    #print(np.sum(samplesPerClass))


def epic_preload_imgs(args):

    baseFolder = './'

    trainF = open('{}epic_train.txt'.format(baseFolder), 'r')
    trainSamples = trainF.readlines()
    trainF.close()

    totalSamples = len(trainSamples)

    trainImgsF = h5py.File('{}epic_train_imgs_{}.h5'.format(baseFolder, args.sequence_length), 'a')
    trainImgs = trainImgsF.create_dataset('data', shape=(totalSamples * (args.sequence_length + 1), 300, 300, 3))

    n = int((args.sequence_length - 1)/2)

    count = 0
    curSam = 0

    for sample in trainSamples:
        print('Train {}/{}'.format(curSam + 1, totalSamples))
        curSam += 1
        entries = sample.strip().split(' ')
        pid = entries[0]
        vid = entries[1]
        frameId = int(entries[2])
        for i in range(frameId - n -1, frameId + n + 1):
            img_path = '{}EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train/{}/{}/'.format(baseFolder, pid, vid) + ('frame_%10d.jpg' % i).replace(' ', '0')
            img = img_to_array(load_img(img_path, target_size=(300, 300)))
            img[:] = img[:] / 127.5 - 1.
            trainImgs[count, ...] = img[:]
            count += 1

    trainImgsF.close()

    testF = open('./epic_test.txt', 'r')
    testSamples = testF.readlines()
    testF.close()

    totalSamples = len(testSamples)

    testImgsF = h5py.File(
        '{}epic_test_imgs_{}.h5'.format(baseFolder, args.sequence_length), 'a')
    testImgs = testImgsF.create_dataset('data', shape=(totalSamples * (args.sequence_length + 1), 300, 300, 3))

    n = int((args.sequence_length - 1) / 2)

    count = 0
    curSam = 0
    for sample in testSamples:
        print('Test {}/{}'.format(curSam + 1, totalSamples))
        curSam += 1
        entries = sample.strip().split(' ')
        pid = entries[0]
        vid = entries[1]
        frameId = int(entries[2])
        for i in range(frameId - n - 1, frameId + n + 1):
            img_path = '{}EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train/{}/{}/'.format(
                baseFolder, pid, vid) + ('frame_%10d.jpg' % i).replace(' ', '0')
            img = img_to_array(load_img(img_path, target_size=(300, 300)))
            img[:] = img[:] / 127.5 - 1.
            testImgs[count, ...] = img[:]
            count += 1

    testImgsF.close()


def extract_from_tars(numT=6):

    baseFolder = './'

    def singleT(pfolderList):
        #print(pfolderList)
        for pfolder in pfolderList:
            fullFolderPath = '{}EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train/{}/'.format(baseFolder, pfolder)
            for tar in os.listdir(fullFolderPath):
                if tar[-4:] != '.tar': continue
                if not os.path.exists(fullFolderPath + tar[:-4]):
                    os.mkdir(fullFolderPath + tar[:-4])
                os.system('tar -xvf {} -C {}'.format(fullFolderPath + tar, fullFolderPath + tar[:-4]))

    import multiprocessing

    pfolders = os.listdir('{}EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train/'.format(baseFolder))
    numPfolders = len(pfolders)

    fInEachT = numPfolders // numT + 1

    processes = []
    for i in range(0, numPfolders, fInEachT):
        p = multiprocessing.Process(target=singleT, args=(pfolders[i:min(i + numPfolders, numPfolders)],))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--option',
                        help='tell it what to do',
                        type=str,
                        default='')
    parser.add_argument('--load_exist_model',
                        help='whether to train from existing models',
                        action='store_true')
    parser.add_argument('--load_from',
                        help='the weights file to load',
                        type=str,
                        default='')
    parser.add_argument('--init_learning_rate',
                        help='specify the initial learning rate',
                        type=float,
                        default=0.003)
    parser.add_argument('--decay',
                        help='specify the decay',
                        type=float,
                        default=0.0001)
    parser.add_argument('--l2_reg',
                        help='specify the value of l2 regularizer',
                        type=float,
                        default=0.00005)
    parser.add_argument('--num_gpu',
                        help='Only relevant during training.Nnum of gpus to use (more than 1 will lead to multi-gpu training)',
                        type=int,
                        default=1)
    parser.add_argument('--batch_size',
                        help='specify the batch size',
                        type=int,
                        default=1)
    parser.add_argument('--patience',
                        help='patience to decay learning rate. if 0 is given, a self-defined scheduler is user',
                        type=int,
                        default=3)
    parser.add_argument('--steps_per_epoch',
                        help='steps per epoch',
                        type=int,
                        default=1000)
    parser.add_argument('--epochs',
                        help='max epochs epoch',
                        type=int,
                        default=80)
    parser.add_argument('--initial_epoch',
                        help='the index of the epoch where the training starts. useful for restoring training from training weights',
                        type=int,
                        default=0)
    parser.add_argument('--take_all',
                        help='whether to take all memory of available GPUs',
                        action='store_true')
    parser.add_argument('--oneHotMode',
                        help='specify the mode of one hot attention labels. 0: pure 0-1 one hot; 1: similarities; 2: softmaxed similarities',
                        type=int,
                        default=0)
    parser.add_argument('--sequence_length',
                        help='specify the length of the sequence (at least 8)',
                        type=int,
                        default=15)
    parser.add_argument('--input_height',
                        help='specify the height of the frame (at least 32)',
                        type=int,
                        default=300)
    parser.add_argument('--input_width',
                        help='specify the width of the frame (at least 32)',
                        type=int,
                        default=300)
    parser.add_argument('--where_help_inside',
                        help='whether where helps inside',
                        action='store_true')
    parser.add_argument('--what_help_inside',
                        help='whether what helps inside',
                        action='store_true')
    parser.add_argument('--vgg_BN_momentum',
                        help='specify the momentum of batch normalization layer for training vgg16 ssd',
                        type=float,
                        default=0.8)
    parser.add_argument('--i3d_BN_momentum',
                        help='specify the momentum of batch normalization layer for training vgg16 ssd',
                        type=float,
                        default=0.8)
    parser.add_argument('--softArgmax',
                        help='whether to use soft argmax in the where_help_inside module',
                        action='store_true')
    parser.add_argument('--num_val_samples',
                        help='specify the number of validation samples, default to the whole val set, however, setting to a smaller number makes the validation step after each epoch faster',
                        type=int,
                        default=13182)

    args = parser.parse_args()

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto(allow_soft_placement=True)
    if not args.take_all:
        #print(args.take_all)
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))


    NUM_CLASS = 53 # This is the number of classes used in rescaleTensor2() and softPredBoxClassification()
                   # It is obtained after we run epic_preprocess()
                   # We directly put it here as a global variable for simplicity
                   # Of course, a more fancy way would be defining rescaleTensor2() and softPredBoxClassification() as classes and initialize them with corresponding class numbe

    if args.option == 'extractepic':
        extract_from_tars()
    elif args.option == 'processepic':
        epic_preprocess()
    elif args.option == 'encodeepic':
        encodeLabelsEpic()
    elif args.option == 'preloadepic':
        epic_preload_imgs(args)
    elif args.option == 'trainepic':
        assert args.softArgmax, 'During training, soft argmax should be used so that gradients can backpropagate properly'
        train(args)
    elif args.option == 'testepic':
        assert args.batch_size == 1, 'During testing batch size must be set to 1'
        test(args)
    else:
        print('Invalid Option!!!')
