'''This function is the operation of transformation'''

import numpy as np
def transformation(array,image_size=0):

    ite = int((image_size/2)+1)
    print(ite)
    for num in range(1, ite):
        array[(num*2)-1] = array[(num*2)-1, ::-1]

    return array
