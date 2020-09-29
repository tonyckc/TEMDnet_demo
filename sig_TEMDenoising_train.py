# -*- coding: utf-8 -*-

"""
Created on Mon Jul 16 00:56:12 2018

@author: kecheng chen
"""

import datetime
import numpy as np
import sys

sys.path.insert(0, './model')
import tensorflow as tf
import os
from PIL import Image
import scipy.misc as misc
import random
from argparse import ArgumentParser
import sig_optimize
import sig_test
import sig_proprocess

training_id = ''
#MODEL_SAVE_PATH = '/home/chenkecheng/TEM_ADMM/trianing_data/{}/logs_{}'.format(training_id, training_id)
MODEL_SAVE_PATH = ''
TENSORBOARD_SAVE_PATH = '/home/chenkecheng/TEM_ADMM/trianing_data/{}/tensorboard_{}'.format(training_id, training_id)
DENOISING_IMG_PATH = '/home/chenkecheng/TEM_ADMM/trianing_data/{}/denoising_img_{}'.format(training_id, training_id)
DATASET_PATH_INPUT = '/hdd/chenkecheng/TEM_ADMM/src/noise_signal.mat'
DATASET_PATH_REAL = '/hdd/chenkecheng/TEM_ADMM/src/clean_signal.mat'
#DATASET_PATH_TEST = '/hdd/chenkecheng/TEM_ADMM/src/test.mat'
DATASET_PATH_TEST = ''
#TEST_SAVE_PATH = ''
TEST_SAVE_PATH = ''
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.98
EPOCHS = 222
LOAD_MODEL = True
TRAIN_MODEL = False
BATCH_SIZE = 128
TEST_BATCH_SIZE= 20
IMG_SIZE = 50
DEVICE = '/gpu:0'
STDDEV = 0.01


def train():
    input_data, real_data, test_data = sig_proprocess.proprocess(MODEL_SAVE_PATH=MODEL_SAVE_PATH,
                                                             TENSORBOARD_SAVE_PATH=TENSORBOARD_SAVE_PATH,
                                                             DENOISING_IMG_PATH=DENOISING_IMG_PATH,
                                                             DATASET_PATH_INPUT=DATASET_PATH_INPUT,
                                                             DATASET_PATH_REAL=DATASET_PATH_REAL,
                                                             DATASET_PATH_TEST=DATASET_PATH_TEST)

    sig_optimize.train(input_data, real_data, test_data, training_id, MODEL_SAVE_PATH, TENSORBOARD_SAVE_PATH,
                   DENOISING_IMG_PATH, LEARNING_RATE_BASE, LEARNING_RATE_DECAY, EPOCHS, LOAD_MODEL=LOAD_MODEL,
                   BATCH_SIZE=BATCH_SIZE,
                   IMG_SIZE=IMG_SIZE, DEVICE=DEVICE, STDDEV=STDDEV)


if __name__ == '__main__':
    if TRAIN_MODEL:
        train()
    else:
        sig_test.evaluate(MODEL_SAVE_PATH=MODEL_SAVE_PATH, BATCH_SIZE=TEST_BATCH_SIZE, SAVE_PATH=TEST_SAVE_PATH,
                      IMG_SIZE=IMG_SIZE, training_id=training_id, STDDEV=STDDEV)
