
import tensorflow as tf
import os
from basic_op import conv_op
from basic_op import res_block_layers_v1
from basic_op import res_block_layers_v2
from basic_op import dilated_conv_op

def denoiser(input, reuse=False, name='denoiser',training=True,STDDEV=0.01):
    with tf.variable_scope(name, reuse= reuse):
        dilated_conv1 = dilated_conv_op(input, 'dilated_conv1', 128, training=training, useBN=False, kh=3, kw=3,
                                        rate=1, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv2 = dilated_conv_op(dilated_conv1, 'dilated_conv2', 128, training=training, useBN=False, kh=3, kw=3,
                                        rate=2, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv3 = dilated_conv_op(dilated_conv2, 'dilated_conv3', 128, training=training, useBN=True, kh=3, kw=3,
                                        rate=3, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv4 = dilated_conv_op(dilated_conv3, 'dilated_conv4', 128, training=training, useBN=True, kh=3, kw=3,
                                        rate=4, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv5 = dilated_conv_op(dilated_conv4, 'dilated_conv5', 128, training=training, useBN=True, kh=3, kw=3,
                                        rate=3, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)
        dilated_conv6 = dilated_conv_op(dilated_conv5, 'dilated_conv7', 128, training=training, useBN=True, kh=3, kw=3,
                                        rate=2, padding="SAME",
                                        activation=tf.nn.relu, STDDEV=STDDEV)

        dilated_conv7 = dilated_conv_op(dilated_conv6, 'dilated_conv8', 1, training=training, useBN=None, kh=3, kw=3,
                                        rate=1, padding="SAME",
                                        activation=None, STDDEV=STDDEV)
        result = dilated_conv7

        return result