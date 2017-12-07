import os
import scipy.misc as scm
import numpy as np

def make_project_dir(project_dir):
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        os.makedirs(os.path.join(project_dir, 'models'))
        os.makedirs(os.path.join(project_dir, 'result'))
        os.makedirs(os.path.join(project_dir, 'result_test'))


def get_image(img_path, data_size):
    img = scm.imread(img_path)
    img_crop = img[15:203,9:169,:]
    img_resize = scm.imresize(img_crop,[data_size,data_size,3])
    img_resize = img_resize/127.5 - 1.
    
    return img_resize


def inverse_image(img):
    img = (img + 1.) * 127.5
    img[img > 255] = 255.
    img[img < 0] = 0.
#    img = img[..., ::-1] # bgr to rgb
    return img.astype(np.uint8)

