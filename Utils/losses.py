import tensorflow as tf
import random
lamb = 100
def signal_loss(output, real, batchsize, image_size):
    output = tf.multiply(tf.add(output, 1), 127.5)
    real = tf.multiply(tf.add(real, 1), 127.5)
    loss = 0
    a = random.sample(range(0,8), 1)
    for num in range(batchsize):
        loss_points = 0
        for i in range(image_size):
            for j in range(image_size):
                loss_points += tf.abs((output[num, j, i, 0]*(256**0) + output[num, j, i, 1]*(256**1) + \
                             output[num, j, i, 1]*(256**1)) - (real[num, j, i, 0]*(256**0) + real[num, j, i, 1]*(256**1) + \
                             real[num, j, i, 1]*(256**1)))

        loss_one_mean = loss_points/(image_size*image_size)
        loss =+ loss_one_mean
    loss_images_mean = loss/batchsize
    return loss_images_mean

def res_loss(input,output, real):
    loss = tf.reduce_mean(tf.square(output - (input-real)))
    return loss
def res_loss_MAE(input,output, real):
    loss = tf.reduce_mean(tf.abs(output - (input-real)))
    return loss
def MSE_loss(output, real):
    loss = tf.reduce_mean(tf.square(output - real))
    return loss