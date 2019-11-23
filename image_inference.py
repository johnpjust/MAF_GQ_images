import os
import json
import pprint
import datetime
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import glob
import random
import matplotlib.pyplot as plt
from skimage import io, transform

from scipy.optimize import fmin_l_bfgs_b

import functools

def img_heatmap(model, sess, imgcre, args):
    # rand_box = np.append(tf.cast(tf.multiply(tf.cast(imgcre.shape[:2], tf.float32),tf.constant(0.1)), tf.int32).numpy(), [3])
    rows = imgcre.shape[0]-args.rand_box[0]
    cols = imgcre.shape[1] - args.rand_box[1]
    heatmap = np.zeros((np.int(rows/args.spacing), np.int(cols/args.spacing)))
    with tf.device(args.device):
        if type(args.vh) is not np.ndarray:
            im_breakup_array = np.zeros((np.int(cols / args.spacing), args.rand_box_size * args.rand_box_size * 3), dtype=np.float32)
            for i in range(np.int(rows/args.spacing)*args.spacing):
                if not i%args.spacing:
                    for j in range(np.int(cols/args.spacing)*args.spacing):
                        if not j%args.spacing:
                            # im_breakup_array[np.int(j/args.spacing),:] = tf.reshape(tf.image.crop_to_bounding_box(imgcre, i, j, args.rand_box_size, args.rand_box_size)/128 - 1, [-1]).numpy()
                            im_breakup_array[np.int(j/args.spacing),:] = (imgcre[i:(i+args.rand_box_size), j:(j+args.rand_box_size)]/128 - 1).reshape([-1])
                    heatmap[np.int(i/args.spacing), :] = model.eval(im_breakup_array, sess)
        else:
            for i in range(np.int(rows/args.spacing)*args.spacing):
                im_breakup_array = np.zeros((np.int(cols / args.spacing), args.vh.shape[1]), dtype=np.float32)
                if not i%args.spacing:
                    for j in range(np.int(cols/args.spacing)*args.spacing):
                        if not j%args.spacing:
                            # im_breakup_array[np.int(j/args.spacing),:] = tf.squeeze(tf.matmul(tf.reshape(tf.image.crop_to_bounding_box(imgcre, i, j, args.rand_box_size, args.rand_box_size)/128 - 1, [1, -1]), args.vh)).numpy()
                            im_breakup_array[np.int(j/args.spacing),:] = np.squeeze(np.matmul((imgcre[i:(i+args.rand_box_size), j:(j+args.rand_box_size)]/128 - 1).reshape([1,-1]), args.vh))
                    heatmap[np.int(i/args.spacing), :] = model.eval(im_breakup_array, sess)
    return heatmap

def img_load(filename, args):
    # img_raw = tf.io.read_file(filename)
    # img = tf.image.decode_image(img_raw)
    img = io.imread(filename)
    offset_width = 50
    offset_height = 10
    target_width = 660
    target_height = 470
    # imgc = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
    imgc = img[offset_height:target_height, offset_width:target_width]
    # # args.img_size = 0.25;  args.preserve_aspect_ratio = True; args.rand_box = 0.1
    imresize_ = np.multiply(imgc.shape[:2],args.img_size)
    # imgcre = tf.image.resize(imgc, size=imresize_)
    imgcre = transform.resize(imgc, output_shape=imresize_.astype(np.int), preserve_range=True)
    return imgcre

class parser_:
    pass

def main():
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    # tf.compat.v1.enable_eager_execution(config=config)

    # tf.config.experimental_run_functions_eagerly(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    args = parser_()
    args.device = '/gpu:0'  # '/gpu:0'
    args.save = r'D:\pycharm_projects\GQC_images_tensorboard\corn_layers1_h12_flows6_resize1.0_boxsize0.1_gated_2019-11-21-12-16-24'
    args.spacing = 8
    args.manualSeed = None
    args.manualSeedw = None

    print('Loading dataset..')
    trainval = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
    train_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in trainval])
    cont_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    cont_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in cont_data])


    ## heatmaps:  change args.spacing closer to 1 to get more spatial resolution on the heatmap
    heat_map_train = []
    heat_map_train.extend(img_heatmap(model, sess, img, args) for img in train_data)
    heatmap_t = np.array(heat_map_train)
    heat_map_cont = []
    heat_map_cont.extend(img_heatmap(model, sess, img, args) for img in cont_data)
    heatmap_c = np.array(heat_map_cont)

    ## summaries
    mean_t = np.mean(heatmap_t.reshape((heatmap_t.shape[0], -1)), axis=1)
    std_t = np.std(heatmap_t.reshape((heatmap_t.shape[0], -1)), axis=1)
    min_t = np.min(heatmap_t.reshape((heatmap_t.shape[0], -1)), axis=1)
    max_t = np.max(heatmap_t.reshape((heatmap_t.shape[0], -1)), axis=1)

    mean_c = np.mean(heatmap_c.reshape((heatmap_c.shape[0], -1)), axis=1)
    std_c = np.std(heatmap_c.reshape((heatmap_c.shape[0], -1)), axis=1)
    min_c = np.min(heatmap_c.reshape((heatmap_c.shape[0], -1)), axis=1)
    max_c = np.max(heatmap_c.reshape((heatmap_c.shape[0], -1)), axis=1)

    totals_t = np.stack([mean_t, std_t, min_t, max_t]).T
    totals_c = np.stack([mean_c, std_c, min_c, max_c]).T

    np.savetxt(args.save + '/totals_t' + '_' + str(args.spacing) + '_' + str(heatmap_t.reshape((heatmap_t.shape[0], -1)).shape[1]) + '.csv', totals_t, delimiter=',')
    np.savetxt(args.save + '/totals_c' + '_' + str(args.spacing) + '_' + str(heatmap_c.reshape((heatmap_c.shape[0], -1)).shape[1]) + '.csv', totals_c, delimiter=',')

    ## heatmap arrays
    np.save(args.save + '/totals_t' + '_' + str(args.spacing) + '_' + str(heatmap_t.reshape((heatmap_t.shape[0], -1)).shape[1]) + '.npy', heatmap_t)
    np.save(args.save + '/totals_c' + '_' + str(args.spacing) + '_' + str(heatmap_c.reshape((heatmap_c.shape[0], -1)).shape[1]) + '.npy', heatmap_c)

if __name__ == '__main__':
    main()

##"C:\Program Files\Git\bin\sh.exe" --login -i

#### tensorboard --logdir=C:\Users\justjo\PycharmProjects\BNAF_tensorflow_eager\tensorboard\checkpoint
## http://localhost:6006/

