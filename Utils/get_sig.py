import os
import scipy.io as sio


def get_img(path, name):
    #print(path)
    #print(name)
    data = sio.loadmat(path)
    load_matrix = data[name]
    return load_matrix


