import os
def create_save_folder(path):
    if os.path.exists(path):
        print('{} exists!\n'.format(path))
    else:
        os.makedirs(path)
        print('{} create sucessfully\n'.format(path))