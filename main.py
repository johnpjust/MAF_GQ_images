import train
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mafs
import struct
import numpy as np
import glob
import scipy.io
import gzip
import functools
import random
from skimage import io, transform

tf.random.set_random_seed(None)

def img_preprocessing(imgcre, args):
    # rand_box = np.append(tf.cast(tf.multiply(tf.cast(imgcre.shape[:2], tf.float32),tf.constant(0.1)), tf.int32).numpy(), [3])
    rand_crop = tf.image.random_crop(imgcre, args.rand_box)
    #### random perturbations
    rand_crop = rand_crop + tf.random.uniform(rand_crop.shape, -0.5, 0.5) ## dequantize
    rand_crop = tf.image.random_flip_left_right(rand_crop)
    rand_crop = tf.image.random_brightness(rand_crop, max_delta=32.0) #32.0 / 255.0
    rand_crop = tf.image.random_saturation(rand_crop, lower=0.5, upper=1.5)
    rand_crop = tf.clip_by_value(rand_crop, 0, 255)
    if type(args.vh) is np.ndarray:
        return tf.squeeze(tf.matmul(tf.reshape(rand_crop/128 - 1, [1,-1]), args.vh.T))
    else:
        return tf.reshape(rand_crop/128 - 1, [-1])

def img_preprocessing_val(imgcre, args):
    rand_box_size = np.int(imgcre.shape[0]*args.rand_box)
    rand_box = np.array([rand_box_size,rand_box_size,3])
    # rand_box = np.append(tf.cast(tf.multiply(tf.cast(imgcre.shape[:2], tf.float32),tf.constant(0.1)), tf.int32).numpy(), [3])
    rand_crop = tf.image.random_crop(imgcre, rand_box)
    if type(args.vh) is np.ndarray:
        return tf.squeeze(tf.matmul(tf.reshape(rand_crop/128 - 1, [1,-1]), args.vh.T))
    else:
        return tf.reshape(rand_crop/128 - 1, [-1])

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

def load_dataset(args):

    tf.random.set_random_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    trainval = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
    train_data = np.vstack([np.expand_dims(img_load(x, args),axis=0) for x in trainval])
    cont_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    cont_data = np.vstack([np.expand_dims(img_load(x,args),axis=0) for x in cont_data])

    args.rand_box_size = np.int(train_data[0].shape[0] * args.rand_box_init)
    args.rand_box = np.array([args.rand_box_size, args.rand_box_size, 3])
    args.n_dims = np.prod(args.rand_box)

    img_preprocessing_train = functools.partial(img_preprocessing, args=args)
    img_preprocessing_val = functools.partial(img_preprocessing, args=args)

    dataset_train = tf.data.Dataset.from_tensor_slices(train_data.astype(np.float32))#.float().to(args.device)
    dataset_train = dataset_train.shuffle(buffer_size=len(train_data)).map(img_preprocessing_train, num_parallel_calls=args.parallel).batch(batch_size=args.batch_size).prefetch(buffer_size=args.prefetch_size)
    # dataset_train = dataset_train.shuffle(buffer_size=len(train)).batch(batch_size=args.batch_size).prefetch(buffer_size=args.prefetch_size)

    if args.vh:
        dataset_train = dataset_train.repeat(args.max_iterations)
        dataset_train = dataset_train.make_one_shot_iterator()
        dataset_train = dataset_train.get_next()
        cliplist = []
        for n in range(20):
            cliplist.append(sess.run(dataset_train))
        svdmat = np.vstack(cliplist)
        _, _, args.vh = scipy.linalg.svd(svdmat, full_matrices=False)
        img_preprocessing_train = functools.partial(img_preprocessing, args=args)
        img_preprocessing_val = functools.partial(img_preprocessing, args=args)
        dataset_train = tf.data.Dataset.from_tensor_slices(train_data.astype(np.float32))  # .float().to(args.device)
        dataset_train = dataset_train.shuffle(buffer_size=len(train_data)).map(img_preprocessing_train,num_parallel_calls=args.parallel).batch(batch_size=args.batch_size).prefetch(buffer_size=args.prefetch_size)

    dataset_valid = tf.data.Dataset.from_tensor_slices(train_data.astype(np.float32))#.float().to(args.device)
    dataset_valid = dataset_valid.map(img_preprocessing_val, num_parallel_calls=args.parallel).batch(batch_size=args.batch_size*2).prefetch(buffer_size=args.prefetch_size)
    # dataset_valid = dataset_valid.batch(batch_size=args.batch_size*2).prefetch(buffer_size=args.prefetch_size)

    dataset_cont = tf.data.Dataset.from_tensor_slices(cont_data.astype(np.float32))#.float().to(args.device)
    dataset_cont = dataset_cont.map(img_preprocessing_val, num_parallel_calls=args.parallel).batch(batch_size=args.batch_size*2).prefetch(buffer_size=args.prefetch_size)

    # args.n_dims = train.shape[1]
    return dataset_train, dataset_valid, dataset_cont

class parser_:
    pass

args = parser_()
args.early_stopping = 100
args.check_every = 5
args.show_log = True
args.batch_size = 500
args.max_iterations = 20000
args.num_layers = 5
args.num_hidden = [100]
args.act = tf.nn.relu
args.vh = 0 #0 =no, 1=yes
args.prefetch_size = 1  # data pipeline prefetch buffer size
args.parallel = 16  # data pipeline parallel processes
args.img_size = 0.25;  ## resize img between 0 and 1
args.preserve_aspect_ratio = True;  ##when resizing
args.rand_box_init = 0.1  ##relative size of random box from image
args.manualSeed = None

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
args.sess = sess

data_loader_train, data_loader_valid, data_loader_cont = load_dataset(args)
### train
train_data = data_loader_train.repeat(args.max_iterations)
train_data = train_data.make_one_shot_iterator()
train_data = train_data.get_next()
### val
val = data_loader_valid.repeat(args.max_iterations)
val = val.make_one_shot_iterator()
val = val.get_next()
### test
cont_data = data_loader_cont.repeat(args.max_iterations)
cont_data = cont_data.make_one_shot_iterator()
cont_data = cont_data.get_next()

## build model
model = mafs.MaskedAutoregressiveFlow(args.n_dims, args.num_hidden, args.act, args.num_layers, batch_norm=True)

t = train.Trainer(model) ## only pass model but don't re-initialize for SCE

## optimizer has some kind of parameters that require initialization
init=tf.global_variables_initializer()
sess.run(init)

## MLE training
t.train(sess, train_data, val, cont_data, early_stopping=args.early_stopping, check_every_N=args.check_every, show_log=args.show_log, max_iterations=args.max_iterations, saver_name='temp/tmp_model')
# t.train(sess, data.astype(np.float32), val_data=val.astype(np.float32), early_stopping=100, check_every_N=5, show_log=True, batch_size=100, max_iterations=20000, saver_name='temp/tmp_model')

# import matplotlib.pyplot as plt
# import scipy.stats
s = model.gen(sess, 5000)
# out = model.eval(train_data, sess)
out2 = model.eval(sess.run(val), sess)
out3 = model.eval(sess.run(cont_data), sess)
sout_ = model.eval(s,sess)
# dist = scipy.stats.johnsonsu.fit(out)
# out = (np.arcsinh((out - dist[-2]) / dist[-1]) * dist[1] + dist[0])
# out2 = (np.arcsinh((out2 - dist[-2]) / dist[-1]) * dist[1] + dist[0])
# out3 = (np.arcsinh((out3 - dist[-2]) / dist[-1]) * dist[1] + dist[0])
# sout = (np.arcsinh((sout_ - dist[-2]) / dist[-1]) * dist[1] + dist[0])

# plt.figure()
# plt.hist(out, 50, density=True, alpha=0.3, label='cifar_train')
# plt.hist(out2, 50, density=True, alpha=0.3, label='cifar_val')
# plt.hist(out3, 50, density=True, alpha=0.3, label='svhn')
# plt.hist(sout, 50, density=True, alpha=0.3, label='samples')
# plt.xlabel('MAF Density')
# plt.legend()
# plt.xlim([-6,3])
# plt.savefig(r'C:\Users\justjo\Desktop\maf_cifarVSsvhn_density_samples_'
#             r'.png', bbox_inches='tight')
#
# saver = tf.train.Saver()
# saver.save(sess, r'C:\Users\justjo\PycharmProjects\maf_tf\Models\maf_digitsVSfashion_h100_f5_tanh_dim784\model')
#
# # s = model.gen(sess, 5000)
# plt.figure();plt.scatter(data[:,0], data[:,1], alpha=1, label='data')
# plt.scatter(test[:,0], test[:,1], alpha=0.2)
# plt.scatter(s[:,0], s[:,1], alpha=0.3, label='sampled')
# st = np.matmul(s, vh)
# plt.scatter(st[:,0], st[:,1], alpha=0.2, label='MAF_SVD')