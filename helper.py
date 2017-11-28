import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.models import load_model
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def maybe_download_training_img(data_dir):
    data_road_filename = 'data_road.zip'
    data_road_path = os.path.join(data_dir, 'data_road')

    if not os.path.exists(data_road_path):
        # Clean dataset dir
        if os.path.exists(data_road_path):
            shutil.rmtree(data_road_path)
        os.makedirs(data_road_path)

        # Download dataset
        print('Downloading dataset...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3.us-east-2.amazonaws.com/hosted-downloadable-files/data_road.zip',
                os.path.join(data_road_path, data_road_filename),
                pbar.hook)

        # Extract dataset
        print('Extracting dataset...')
        zip_ref = zipfile.ZipFile(os.path.join(data_road_path, data_road_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(data_road_path, data_road_filename))

def gen_test_output(model, data_folder, image_shape):

    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)


        im_softmax = model.predict(np.array([image]))
        im_softmax = im_softmax[0][:, :, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)

        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im), segmentation

def save_inference_samples(runs_dir, data_dir, image_shape, model_path):
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    model = load_model(model_path)

    model.evaluate()
    image_outputs = gen_test_output(model, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image, seg in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        np.save(os.path.join(output_dir, name), seg)

def get_data(data_dir, image_shape):
    image_paths = glob(os.path.join(data_dir, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_dir, 'gt_image_2', '*_road_*.png'))}
    background_color = np.array([255, 0, 0])

    images = []
    gt_images = []
    for image_file in image_paths:
        gt_image_file = label_paths[os.path.basename(image_file)]

        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

        images.append(image)
        gt_images.append(gt_image)

    return np.array(images), np.array(gt_images)