import tensorflow as tf
import os
from basic_op import conv_op
from basic_op import res_block_layers_v1
from basic_op import res_block_layers_v2
from basic_op import dilated_conv_op

def denoiser(input, reuse=False, name='denoiser',training=True,STDDEV=0.01):
    with tf.variable_scope(name, reuse= reuse):
        conv1 = conv_op(input, name + "_localConv1", 64, training=training, useBN=None, kh=3, kw=3,
                dh=1, dw=1,padding="SAME", activation=tf.nn.relu, STDDEV=STDDEV)
        conv2 = conv_op(conv1, name + "_localConv2", 64, training=training, useBN=True, kh=3, kw=3,
                        dh=1, dw=1, padding="SAME", activation=tf.nn.relu, STDDEV=STDDEV)
        conv_result = []
        conv_result.append(conv2)
        for num in range(1,14):
            conv_result.append(conv_op(conv_result[num-1], name + "_localConv"+str(num+2), 64, training=training, useBN=True, kh=3, kw=3,
                        dh=1, dw=1, padding="SAME", activation=tf.nn.relu, STDDEV=STDDEV))
        conv13 = conv_op(conv_result[12], name + "_localConv13", 64, training=training, useBN=True, kh=3, kw=3,
                            dh=1, dw=1, padding="SAME", activation=tf.nn.relu, STDDEV=STDDEV)
        conv14 = conv_op(conv13, name + "_localConv14", 64, training=training, useBN=True, kh=3, kw=3,
                         dh=1, dw=1, padding="SAME", activation=tf.nn.relu, STDDEV=STDDEV)
        conv15 = conv_op(conv14, name + "_localConv15", 1, training=training, useBN=None, kh=3, kw=3,
                         dh=1, dw=1, padding="SAME", activation=None, STDDEV=STDDEV)
        return conv15