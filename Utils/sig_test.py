# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:17:18 2019

@author: ckc
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys
sys.path.insert(0, './model')
import  datetime
from sig_noise_prior import model as net
from plainCNN_noise_prior import denoiser
from sig_noise_prior_nodilated import model as net2
from sig_noise_prior_nores import model as net3
from sig_noise_prior_res import model as net4
from plainCNN_noise_prior import denoiser
from sig_noise_prior_nores_v3 import model as net5
from Resnet_9 import model as net6
from Resnet_6 import model as net7
from FFDNet import denoiser as net8
from IRCNN import denoiser as net9
from DnCNN import denoiser as net10
from get_sig import get_img
from transformation import transformation as trans
from get_next_batch_sig import get_next_batch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DATASET_PATH_TEST = './src/test'
import scipy.io as scio
test_id=6
def evaluate(name=False,MODEL_SAVE_PATH=None, BATCH_SIZE=2, SAVE_PATH=None, IMG_SIZE=0, training_id=None, STDDEV=None, is_TEMDNet=False):
    #print(IMG_SIZE)
    #print(MODEL_SAVE_PATH)
    tf.reset_default_graph()
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print("make success!")
    checkpoint_dir = MODEL_SAVE_PATH
    print(checkpoint_dir)
    z_placeholder = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1], name='z_placeholder')

    if is_TEMDNet:
        generated_images = net(z_placeholder, reuse=False, name='nosie_prior', training=False, STDDEV=STDDEV)
        print('TEMD')
    else:
        if name=='DnCNN':
            generated_images = net10(z_placeholder, reuse=False, name='nosie_prior', training=False, STDDEV=STDDEV)
        if name=='IRCNN':
            generated_images = net9(z_placeholder, reuse=False, name='nosie_prior', training=False, STDDEV=STDDEV)
        if name=='FFDNet':
            generated_images = net8(z_placeholder, reuse=False, name='nosie_prior', training=False, STDDEV=STDDEV)
        if name=='Res6':
            generated_images = net7(z_placeholder, reuse=False, name='nosie_prior', training=False, STDDEV=STDDEV)
        if name=='Res9':
            generated_images = net6(z_placeholder, reuse=False, name='nosie_prior', training=False, STDDEV=STDDEV)
        if name == 'Unet':
                generated_images = net4(z_placeholder, reuse=False, name='nosie_prior', training=False, STDDEV=STDDEV)
        if name=='Notrans':
            generated_images = net(z_placeholder, reuse=False, name='nosie_prior', training=False, STDDEV=STDDEV)
            print('Notrans')
    #generated_images = denoiser(z_placeholder, reuse=False, name='denoiser')
    print("Reading checkpoints ...")
    saver = tf.train.Saver()  # tf.global_variables()
    # generated_images = net(z_placeholder, reuse=False, name='nosie_prior', training=False)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:

            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print("Loading success,global_step is %s" % global_step)
            print("%s" % (datetime.datetime.now()))
            images = sess.run(generated_images, feed_dict={z_placeholder: get_next_batch(get_img(DATASET_PATH_TEST, 'test'), 0,
                                                                                                 BATCH_SIZE, IMG_SIZE=IMG_SIZE,is_TEMDNet=is_TEMDNet)})
            print("%s" % (datetime.datetime.now()))
            tmp = get_next_batch(get_img(DATASET_PATH_TEST, 'test'), 0, BATCH_SIZE,IMG_SIZE=IMG_SIZE,is_TEMDNet=is_TEMDNet)
            img_id = 0
            output = np.zeros((BATCH_SIZE,IMG_SIZE*IMG_SIZE))
            noise_output = np.zeros((BATCH_SIZE,IMG_SIZE*IMG_SIZE))
            for img in images[0:BATCH_SIZE, :, :, :]:
                    noise = np.reshape(img, (IMG_SIZE, IMG_SIZE, 1))
                    noised_image = tmp[img_id]
                    denoise_image = (noised_image - noise)
                    if is_TEMDNet:
                        #print('trans')
                        denoise_image = trans(denoise_image, IMG_SIZE)
                        denoise_image = np.reshape(denoise_image,(1,IMG_SIZE*IMG_SIZE))
                        noise = trans(noise, IMG_SIZE)
                        noise = np.reshape(noise,(1,IMG_SIZE*IMG_SIZE))
                    else:

                        denoise_image = np.reshape(denoise_image, (1, IMG_SIZE * IMG_SIZE))
                        noise = np.reshape(noise, (1, IMG_SIZE * IMG_SIZE))

                    output[img_id] = denoise_image
                    noise_output[img_id] = noise
                    denoise_sig_name = '{}_{}_{}'.format(name,test_id, IMG_SIZE*IMG_SIZE)
                    noise_name = 'noise_{}_{}_{}'.format(name,test_id, IMG_SIZE*IMG_SIZE)
                    '''
                    if name=='FFDNet':
                        noise_name = '{}_{}_{}'.format(name, test_id, IMG_SIZE * IMG_SIZE)
                    if name == 'DnCNN':
                        noise_name = '{}_{}_{}'.format(name, test_id, IMG_SIZE * IMG_SIZE)
                    '''
                    save_path = SAVE_PATH + '//{}_{}_{}.mat'.format(name,test_id, IMG_SIZE*IMG_SIZE)
                    save_path_sig = SAVE_PATH + '//noise_{}_{}_{}.mat'.format(name,test_id, IMG_SIZE*IMG_SIZE)
                    '''
                    if name == 'FFDNet':
                        save_path_sig = SAVE_PATH + '//{}_{}_{}.mat'.format(name, test_id, IMG_SIZE * IMG_SIZE)
                    if name == 'DnCNN':
                        save_path_sig = SAVE_PATH + '//{}_{}_{}.mat'.format(name, test_id, IMG_SIZE * IMG_SIZE)
                    '''
                    img_id += 1
            print('Test Images are Saved\n')
            print(output.shape)

            scio.savemat(save_path, {denoise_sig_name: output})
            scio.savemat(save_path_sig, {noise_name: noise_output})