import os
import scipy.misc as scm

def make_project_dir(project_dir):
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        os.makedirs(os.path.join(project_dir, 'models'))
        os.makedirs(os.path.join(project_dir, 'result'))
        os.makedirs(os.path.join(project_dir, 'result_test'))


def get_image(img_path, data_size):
    img = scm.imread(img_path)/127.5 - 1.
    img_crop = img[15:203,9:169,:]
    img_resize = scm.imresize(img_crop,[data_size,data_size,3])
#    img = img[..., ::-1]  # rgb to bgr
    return img_resize


def inverse_image(img):
    img = (img + 1.) * 127.5
    img[img > 255] = 255
    img[img < 0] = 0
#    img = img[..., ::-1] # bgr to rgb
    return img

