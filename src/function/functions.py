import os
import scipy.misc as scm
import glob

def make_project_dir(project_dir):
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        os.makedirs(os.path.join(project_dir, 'models'))
        os.makedirs(os.path.join(project_dir, 'result'))
        os.makedirs(os.path.join(project_dir, 'result_test'))


def get_image(img_path):
    img = scm.imread(img_path)/255. - 0.5
    img = img[..., ::-1]  # rgb to bgr
    return img


def inverse_image(img):
    img = (img + 0.5) * 255.
    img[img > 255] = 255
    img[img < 0] = 0
    img = img[..., ::-1] # bgr to rgb
    return img


def load_data(path):
    lst = glob.glob(path)
    image_list = []
    exception = []
    for img in lst:
        try: 
            image_list.append(scm.imread(img))
        except:
            exception.append(img)
#    image_list = [try:scm.imread(img) except:pass for img in lst]
    return image_list, exception