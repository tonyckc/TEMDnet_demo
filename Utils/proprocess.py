from create_save_folder import create_save_folder
from get_img import get_img


def proprocess(MODEL_SAVE_PATH=None,TENSORBOARD_SAVE_PATH=None, DENOISING_IMG_PATH=None, DATASET_PATH_INPUT=None,
               DATASET_PATH_REAL=None, DATASET_PATH_TEST=None):
    create_save_folder(MODEL_SAVE_PATH)
    create_save_folder(TENSORBOARD_SAVE_PATH)
    create_save_folder(DENOISING_IMG_PATH)
    input_data = get_img(DATASET_PATH_INPUT)
    real_data = get_img(DATASET_PATH_REAL)
    test_data = get_img(DATASET_PATH_TEST)
    return input_data,real_data,test_data