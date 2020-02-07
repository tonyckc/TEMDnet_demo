'''The model of Resnet 9'''

import tensorflow as tf
import os
from basic_op import conv_op
from basic_op import res_block_layers_v1
from basic_op import res_block_layers_v2
from basic_op import dilated_conv_op

def model(input, reuse=False, name='nosie_prior', training=True, STDDEV=None):
    with tf.variable_scope(name, reuse=reuse):
        dilated_conv1 = res_block_layers_v1(input, 'block_7', [1, 32], change_dimension=True, block_stride=1,
                                            training=training, STDDEV=STDDEV)
        dilated_conv2 = res_block_layers_v1(dilated_conv1, 'block_8', [32, 64], change_dimension=True, block_stride=1,
                                            training=training, STDDEV=STDDEV)
        '''feature learning '''
        res_block_1 = res_block_layers_v1(dilated_conv2, 'block_1', [64, 128], change_dimension=True, block_stride=1,
                                          training=training, STDDEV=STDDEV)
        res_block_2 = res_block_layers_v2(res_block_1, 'block_2', 128, block_stride=1, training=training, STDDEV=STDDEV)

        dilated_conv3 = res_block_layers_v1(res_block_2, 'block_9', [64, 32], change_dimension=True, block_stride=1,
                                            training=training, STDDEV=STDDEV)
        dilated_conv4 = res_block_layers_v1(dilated_conv3, 'block_10', [32, 1], change_dimension=True, block_stride=1,
                                            training=training, STDDEV=STDDEV, activate=False)
        result = dilated_conv4

        return result