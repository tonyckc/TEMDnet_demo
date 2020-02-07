"""This file is the noise  prior model that conduct preliminary extracting of noise."""
'This model have four dilate convs and four res-blocks '

import tensorflow as tf
import os
from basic_op import conv_op
from basic_op import res_block_layers_v1
from basic_op import res_block_layers_v2
from basic_op import dilated_conv_op


def model(input, reuse=False, name='nosie_prior', training=True, STDDEV=None):
    #-tf.reset_default_graph()
    with tf.variable_scope(name, reuse=reuse):
        dilated_conv1 = dilated_conv_op(input, 'dilated_conv1', 32, training=training, useBN=False, kh=3, kw=3,
                                        rate=1, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv2 = dilated_conv_op(dilated_conv1, 'dilated_conv2', 64, training=training, useBN=False, kh=3, kw=3,
                                        rate=2, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        '''feature learning '''
        res_block_1 = res_block_layers_v1(dilated_conv2, 'block_1', [64, 128], change_dimension=True, block_stride=1,
                                          training=training, STDDEV=STDDEV)
        res_block_2 = res_block_layers_v2(res_block_1, 'block_2', 128, block_stride=1, training=training, STDDEV=STDDEV)
        res_block_3 = res_block_layers_v2(res_block_2, 'block_3', 128, block_stride=1, training=training, STDDEV=STDDEV)
        res_block_4 = res_block_layers_v2(res_block_3, 'block_4', 128, block_stride=1, training=training, STDDEV=STDDEV)
        res_block_5 = res_block_layers_v1(res_block_4, 'block_5', [128, 64], change_dimension=True, block_stride=1,
                                          training=training, STDDEV=STDDEV)

        dilated_conv3 = dilated_conv_op(res_block_5, 'dilated_conv3', 32,  training=training, useBN=False, kh=3, kw=3,
                                        rate=2, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv4 = dilated_conv_op(dilated_conv3, 'dilated_conv4', 1, training=training, useBN=False, kh=3, kw=3,
                                        rate=1, padding="SAME",
                                        activation=None, STDDEV=STDDEV)
        result = dilated_conv4

        return result
