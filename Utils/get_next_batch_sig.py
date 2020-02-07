import numpy as np
from PIL import Image
from transformation import transformation as trans

def get_next_batch(input_img, pointer, batch_size, IMG_SIZE=0, test=False,is_TEMDNet=False):
    #print(IMG_SIZE)
    img_batch = []
    imgs = input_img[pointer * batch_size:(pointer + 1) * batch_size]

    for img in imgs:

        array = np.array(img)
        array = array.reshape((IMG_SIZE, IMG_SIZE, 1))
        if is_TEMDNet:
            #print('batch')
            array = trans(array, IMG_SIZE)
        img_batch.append(array)

    return img_batch