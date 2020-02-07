import datetime
import numpy as np
import sys
sys.path.append('/home/chenkecheng/TEM_ADMM/model')
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6" # notice! set the environment variable to
from PIL import Image
import scipy.misc as misc
import scipy.io as scio
import random
import get_next_batch_sig
from argparse import ArgumentParser
from sig_noise_prior import model as net
from sig_noise_prior_nodilated import model as net2
from sig_noise_prior_nores import model as net3
from sig_noise_prior_res import model as net4
from sig_noise_prior_nores_v3 import model as net5
from Resnet_9 import model as net6
from Resnet_6 import model as net7
from FFDNet import denoiser as net8
from IRCNN import denoiser as net9
from DnCNN import denoiser as net10
#from plainCNN_noise_prior import denoiser
from losses import res_loss
from losses import MSE_loss
from losses import res_loss_MAE

def train(input_data, real_data, test_data, training_id, MODEL_SAVE_PATH, TENSORBOARD_SAVE_PATH,
          DENOISING_IMG_PATH, LEARNING_RATE_BASE, LEARNING_RATE_DECAY, EPOCHS, LOAD_MODEL=False,
          BATCH_SIZE=None, IMG_SIZE=None, DEVICE=None, STDDEV=None):
    global_steps = tf.Variable(initial_value=0, name=
    'global_steps', trainable=False)
    input_placeholder = tf.placeholder(dtype=tf.float32
                                       , shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input_placehloder')
    real_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 1], name='real_placeholder')
    with tf.device(DEVICE):
        #net_output = net8(input_placeholder, reuse=False, name='nosie_prior', STDDEV=STDDEV)
        net_output = net6(input_placeholder, reuse=False, name='nosie_prior',training=True, STDDEV=STDDEV)
        # net_output_test = net(input_placeholder, reuse=False, name='nosie_prior')
        with tf.name_scope('losses'):
            # losses = tf.losses.mean_squared_error(real_placeholder, net_output)
            #losses = MSE_loss(net_output, real_placeholder)
            losses = res_loss(input_placeholder,net_output, real_placeholder)
            tf.summary.scalar('loss', losses)
        with tf.name_scope('learning_rate'):
            learning_rate = tf.train.exponential_decay(
                LEARNING_RATE_BASE,
                global_steps,
                (len(input_data) // BATCH_SIZE),
                LEARNING_RATE_DECAY,
                staircase=True
            )
            tf.summary.scalar('learning_rate', learning_rate)

        saver = tf.train.Saver()
        trainer = tf.train.AdamOptimizer(learning_rate).minimize(losses, global_steps)

        merged = tf.summary.merge_all()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)

        #config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(TENSORBOARD_SAVE_PATH, sess.graph)
            start = 0
            if LOAD_MODEL:
                print("Reading checkpoints...")
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                    saver.restore(sess, os.path.join(MODEL_SAVE_PATH, ckpt_name))
                    global_steps = ckpt.model_checkpoint_path.split('/')[-1] \
                        .split('-')[-1]
                    print("Loading success,global_step is % s" % (global_steps))
                start = int(global_steps)
            steps = start

            print('Start Training....\n')
            '''please input the '''
            for epoch in range(EPOCHS):
                rand_list = random.sample(range(0, len(input_data) // BATCH_SIZE), len(input_data) // BATCH_SIZE)
                for num in range(len(input_data) // BATCH_SIZE):
                    rand_point = rand_list[num]

                    train_ = sess.run(trainer, feed_dict={input_placeholder: get_next_batch_sig.get_next_batch(input_data,
                                                                                            rand_point, BATCH_SIZE,IMG_SIZE),
                                                          real_placeholder: get_next_batch_sig.get_next_batch(real_data, rand_point,
                                                                                           BATCH_SIZE,IMG_SIZE)})
                    print("Epoch: %d, Iteration: %d, At: %s" % (
                        epoch, num, datetime.datetime.now()))
                    if num % 20 == 0:
                        # five iterations write the summary and output the losses
                        summary, loss, _ = sess.run([merged, losses, learning_rate],
                                                    feed_dict={input_placeholder: get_next_batch_sig.get_next_batch(input_data, rand_point,
                                                                                                 BATCH_SIZE,IMG_SIZE),
                                                               real_placeholder: get_next_batch_sig.get_next_batch(real_data, rand_point,
                                                                                                BATCH_SIZE,IMG_SIZE)})
                        writer.add_summary(summary, global_step=steps)

                        print(
                            "Epoch: %d, Iteration: %d, Loss: %f, Learning-rate: %f, At: %s" % (
                            epoch, num, loss, _, datetime.datetime.now()))
                    '''  
                    if num % 200 == 0:
                        print("Start test.....\n")

                        test = sess.run(net_output,
                                        feed_dict={input_placeholder: get_next_batch_sig.get_next_batch(test_data, 0, 10)})
                        tmp = get_next_batch_sig.get_next_batch(test_data, 0, 10)
                        #channel_3 = get_next_batch.get_next_batch(test_data, 0, 10, test=True)
                        img_id = 0
                        for img in test[0:10, :, :, :]:
                            noise = np.reshape(img, (IMG_SIZE, IMG_SIZE, 1))
                            noised_image = tmp[img_id]
                            denoise_image = (noised_image - noise)
                            denoise_image = np.reshape(denoise_image, (1, IMG_SIZE*IMG_SIZE))
                            save_path = DENOISING_IMG_PATH + '//{}_{}_{}.mat'.format(training_id, steps, img_id)
                            scio.savemat(save_path, {'denoise_sig': denoise_image})
                            img_id += 1
                        print('Test Images are Saved\n')
                    '''
                    if num % 200 == 0:
                        path = MODEL_SAVE_PATH + os.path.sep + 'model.ckpt'
                        saver.save(sess, path, global_step=steps)
                    steps += 1
