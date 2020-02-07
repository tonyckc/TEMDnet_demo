import tensorflow as tf

def denoiser(input, reuse=False, name='denoiser'):
    with tf.variable_scope(name, reuse= reuse):
            # conv layer1
            w1 = tf.get_variable('w1', shape=[3, 3, 1, 32], dtype=tf.float32, initializer=
                                 tf.truncated_normal_initializer(stddev=0.01))
            b1 = tf.get_variable('b1', shape=[32], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            conv1 = tf.nn.conv2d(input, w1, strides=[1, 1, 1, 1], padding='SAME')
            conv1 = tf.nn.bias_add(conv1, b1)
            conv1_Re = tf.nn.relu(conv1)

            # conv layer2
            w2 = tf.get_variable('w2', shape=[3, 3, 32, 64], dtype=tf.float32, initializer=
                                 tf.truncated_normal_initializer(stddev=0.01))
            b2 = tf.get_variable('b2', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            conv2 = tf.nn.conv2d(conv1_Re, w2, strides=[1, 1, 1, 1], padding='SAME')
            conv2 = tf.nn.bias_add(conv2, b2)
            conv2_BN = tf.contrib.layers.batch_norm(conv2, epsilon=1e-5, scope='bn1')
            conv2_Re = tf.nn.relu(conv2_BN)

            # conv layer3
            w3 = tf.get_variable('w3', shape=[3, 3, 64, 128], dtype=tf.float32, initializer=
            tf.truncated_normal_initializer(stddev=0.01))
            b3 = tf.get_variable('b3', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            conv3 = tf.nn.conv2d(conv2_Re, w3, strides=[1, 1, 1, 1], padding='SAME')
            conv3 = tf.nn.bias_add(conv3, b3)
            conv3_BN = tf.contrib.layers.batch_norm(conv3, epsilon=1e-5, scope='bn2')
            conv3_Re = tf.nn.relu(conv3_BN)

            # conv layer4
            w4 = tf.get_variable('w4', shape=[3, 3, 128, 128], dtype=tf.float32, initializer=
            tf.truncated_normal_initializer(stddev=0.01))
            b4 = tf.get_variable('b4', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            conv4 = tf.nn.conv2d(conv3_Re, w4, strides=[1, 1, 1, 1], padding='SAME')
            conv4 = tf.nn.bias_add(conv4, b4)
            conv4_BN = tf.contrib.layers.batch_norm(conv4, epsilon=1e-5, scope='bn3')
            conv4_Re = tf.nn.relu(conv4_BN)

            # conv layer5
            w5 = tf.get_variable('w5', shape=[3, 3, 128, 128], dtype=tf.float32, initializer=
            tf.truncated_normal_initializer(stddev=0.01))
            b5 = tf.get_variable('b5', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            conv5 = tf.nn.conv2d(conv4_Re, w5, strides=[1, 1, 1, 1], padding='SAME')
            conv5 = tf.nn.bias_add(conv5, b5)
            conv5_BN = tf.contrib.layers.batch_norm(conv5, epsilon=1e-5, scope='bn4')
            conv5_Re = tf.nn.relu(conv5_BN)

            # conv layer6
            w6 = tf.get_variable('w6', shape=[3, 3, 128, 128], dtype=tf.float32, initializer=
            tf.truncated_normal_initializer(stddev=0.01))
            b6 = tf.get_variable('b6', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            conv6 = tf.nn.conv2d(conv5_Re, w6, strides=[1, 1, 1, 1], padding='SAME')
            conv6 = tf.nn.bias_add(conv6, b6)
            conv6_BN = tf.contrib.layers.batch_norm(conv6, epsilon=1e-5, scope='bn5')
            conv6_Re = tf.nn.relu(conv6_BN)

            # conv layer7
            w7 = tf.get_variable('w7', shape=[3, 3, 128, 128], dtype=tf.float32, initializer=
            tf.truncated_normal_initializer(stddev=0.01))
            b7 = tf.get_variable('b7', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            conv7 = tf.nn.conv2d(conv6_Re, w7, strides=[1, 1, 1, 1], padding='SAME')
            conv7 = tf.nn.bias_add(conv7, b7)
            conv7_BN = tf.contrib.layers.batch_norm(conv7, epsilon=1e-5, scope='bn6')
            conv7_Re = tf.nn.relu(conv7_BN)

            # conv layer8
            w8 = tf.get_variable('w8', shape=[3, 3, 128, 64], dtype=tf.float32, initializer=
            tf.truncated_normal_initializer(stddev=0.01))
            b8 = tf.get_variable('b8', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            conv8 = tf.nn.conv2d(conv7_Re, w8, strides=[1, 1, 1, 1], padding='SAME')
            conv8 = tf.nn.bias_add(conv8, b8)
            conv8_BN = tf.contrib.layers.batch_norm(conv8, epsilon=1e-5, scope='bn7')
            conv8_Re = tf.nn.relu(conv8_BN)

            # conv layer9
            w9 = tf.get_variable('w9', shape=[3, 3, 64, 32], dtype=tf.float32, initializer=
            tf.truncated_normal_initializer(stddev=0.01))
            b9 = tf.get_variable('b9', shape=[32], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            conv9 = tf.nn.conv2d(conv8_Re, w9, strides=[1, 1, 1, 1], padding='SAME')
            conv9 = tf.nn.bias_add(conv9, b9)
            conv9_BN = tf.contrib.layers.batch_norm(conv9, epsilon=1e-5, scope='bn8')
            conv9_Re = tf.nn.relu(conv9_BN)

            # conv layer10
            w10 = tf.get_variable('w10', shape=[3, 3, 32, 1], dtype=tf.float32, initializer=
            tf.truncated_normal_initializer(stddev=0.01))
            b10 = tf.get_variable('b10', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            conv10 = tf.nn.conv2d(conv9_Re, w10, strides=[1, 1, 1, 1], padding='SAME')
            conv10 = tf.nn.bias_add(conv10, b10)
            return conv10
