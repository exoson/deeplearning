import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imresize
import os
import sys
import progressbar
import glob
import csv
import time
import shutil
import cv2

from models import build_generator, build_discriminator, build_deconv_gen, build_conv_discriminator, generator_32, disc_32, discriminator, generator_64
from utils import get_image

img_height = 32
img_width = 32
channels = 1
img_size = img_height * img_width * channels
img_shape = (channels, img_height, img_width)

to_train = True
to_restore = False
output_path = "faijonii"

max_epoch = 1000

z_size = 100
batch_size = 128
beta1 = 0.5
learning_rate = 2e-4
z_shape = (batch_size, z_size)
def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    """
    alpha = 0.2

    with tf.variable_scope('generator', reuse=False if is_train==True else True):
        # First fully connected layer
        x_1 = tf.layers.dense(z, 2*2*512)

        # Reshape it to start the convolutional stack
        deconv_2 = tf.reshape(x_1, (-1, 512, 2, 2))
        batch_norm2 = tf.layers.batch_normalization(deconv_2, training=is_train, axis=1)
        lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)

        # Deconv 1
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 256, 5, 2, padding='VALID', data_format='channels_first')
        batch_norm3 = tf.layers.batch_normalization(deconv3, training=is_train, axis=1)
        lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)


        # Deconv 2
        deconv4 = tf.layers.conv2d_transpose(lrelu3, 128, 5, 2, padding='SAME', data_format='channels_first')
        batch_norm4 = tf.layers.batch_normalization(deconv4, training=is_train, axis=1)
        lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)

        # Output layer
        logits = tf.layers.conv2d_transpose(lrelu4, out_channel_dim, 5, 2, padding='SAME', data_format='channels_first')

        out = tf.tanh(logits)

        return out


def view_imgs(batch):
    k = 0
    while True:
        cv2.imshow("suus", batch[k])
        key = cv2.waitKey(100)
        if key == ord('d'):
            k += 1
        if key == ord('a'):
            k -= 1
        if key == ord('q'):
            break

def show_result(batch_res, grid_size=(4, 4), grid_pad=5, scale=3):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width, channels)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w, channels), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    img = cv2.resize(img_grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Numbers", img)


def mnist_iterator(batch_size):
    import cPickle, gzip
    with gzip.open('../datasets/mnist.pkl.gz') as f:
        (tr_data, _), (vl_data, _), (ts_data, _) = cPickle.load(f)
    data = np.concatenate([tr_data, vl_data, ts_data])
    data = data.reshape(-1, 28, 28, 1)
    data = 2 * data - 1
    new_data = np.zeros((data.shape[0], 1, img_height, img_width))
    for i, sample in enumerate(data):
        new_data[i] = cv2.resize(sample, dsize=(img_height, img_width), interpolation=cv2.INTER_CUBIC).reshape(1, img_height, img_width)
    #data = np.pad(data, pad_width=((0, 0), (0, 0), (2, 2), (2, 2)),
    #              mode='constant', constant_values=-1)

    data = new_data
    if channels == 3:
        data = np.concatenate([data, data, data], axis=1)
    batch_amt = data.shape[0] // batch_size
    i = 0
    yield data.shape[0]
    while True:
        yield data[i*batch_size:(i+1) * batch_size]
        i += 1
        i %= batch_amt


def celeb_iterator(batch_size):
    fnames = glob.glob('../datasets/celebA/*')
    #data = np.array([
    #    get_image(fname, img_size=(64, 64)) for fname in fnames
    #])
    #data = data.transpose(0, 3, 1, 2)
    yield len(fnames)
    batch_amt = len(fnames) // batch_size
    i = 0
    while True:
        batch_files = fnames[i*batch_size:(i+1) * batch_size]
        batch = np.array([
            get_image(fname,
                      input_height=108,
                      input_width=108,
                      resize_height=img_height,
                      resize_width=img_width)[...,::-1]  for fname in batch_files
        ])
        #view_imgs(batch)
        batch = batch.transpose(0, 3, 1, 2)
        yield batch
        i += 1
        i %= batch_amt


def face_iterator(batch_size):
    with file('../datasets/face/training.csv') as fo:
        reader = csv.reader(fo)
        train = list(reader)
    train = train[1:]
    with file('../datasets/face/test.csv') as fo:
        reader = csv.reader(fo)
        test = list(reader)
    test = test[1:]
    data = train + test
    data = np.array([np.fromstring(image[-1], sep=' ') for image in data])
    data /= 255
    data = 2 * data - 1
    i = 0
    data = data.reshape(data.shape[0], 96, 96, 1)
    data = np.concatenate([data, data, data], axis=3)
    data = np.array([imresize(img, (img_width, img_height)) for img in data])
    data = data.transpose(0, 3, 1, 2)
    batch_amt = data.shape[0] // batch_size
    yield data.shape[0]
    while True:
        yield data[i*batch_size:(i+1) * batch_size]
        i += 1
        i %= batch_amt


def data_iterator(filename, batch_size):
    data = np.genfromtxt(filename, delimiter=',')
    data = data[1:, 1:]
    data /= 255
    data = 2 * data - 1
    data = data.reshape(data.shape[0], 28, 28)
    new_data = np.zeros((data.shape[0], 1, img_height, img_width))
    for i, sample in enumerate(data):
        new_data[i] = cv2.resize(sample, dsize=(img_height, img_width), interpolation=cv2.INTER_CUBIC).reshape(1, img_height, img_width)
    data = new_data
    if channels == 3:
        data = np.concatenate([data, data, data], axis=1)
    batch_amt = data.shape[0] // batch_size
    yield data.shape[0]
    i = 0
    while True:
        yield data[i*batch_size:(i+1) * batch_size]
        i += 1
        i %= batch_amt


def catdog_iterator(batch_size):
    import pickle
    with open('../datasets/catdogall.data') as fo:
        data = pickle.load(fo)
    data = np.array(data, dtype=np.float32)
    data /= 255
    data = data.transpose(0, 3, 1, 2)
    data = 2 * data - 1
    i = 0
    batch_amt = data.shape[0] // batch_size
    yield data.shape[0]
    while True:
        yield data[i*batch_size:(i+1) * batch_size]
        i += 1
        i %= batch_amt


def cifar_iterator(batch_size):
    import cPickle
    data = None
    for i in range(1, 5):
        with open('../datasets/cifar-10-batches-py/data_batch_' + str(i), 'rb') as fo:
            data_dict = cPickle.load(fo)
        if data is None:
            data = data_dict['data']
        else:
            data = np.concatenate([data, data_dict['data']])
    data = data.reshape(data.shape[0], 1, 3072)
    data = np.concatenate([data[:, :, :1024], data[:, :, 1024:2048], data[:, :, 2048:]], axis=1)
    data = np.array(data, dtype=np.float32)
    data /= 255
    data = data.reshape((data.shape[0], channels, img_height, img_width))
    data = 2 * data - 1
    #mean = np.mean(data)
    #data = np.subtract(data, mean)
    #std = np.std(data)
    #data = np.divide(data, std)
    i = 0
    yield data.shape[0]
    batch_amt = data.shape[0] // batch_size
    while True:
        yield data[i*batch_size:(i+1) * batch_size]
        i += 1
        i %= batch_amt

def toy_iterator(batch_size):
    yield 100000*batch_size
    while True:
        yield -np.ones((batch_size, channels, img_height, img_width))

def train():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    with tf.device('/gpu:0'):
        x_data = tf.placeholder(tf.float32, [batch_size, channels, img_height, img_width], name="x_data")
        z_prior = tf.placeholder(tf.float32, z_shape, name="z_prior")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        #x_generated = generator(z_prior, channels)
        #x_generated, g_params = build_generator(z_prior, [150, 300, 32*32])
        x_generated,  g_params = generator_32(z_prior, gray_scale=channels==1)
        #x_sample, _ = generator_32(z_prior, train=False, gray_scale=True)
        #x_generated,  g_params = build_deconv_gen(z_prior)
        #x_sample, _ = build_deconv_gen(z_prior, train=False)
        #y_generated, d_params = build_discriminator(x_generated, keep_prob)
        #y_data, _ = build_discriminator(x_data, keep_prob)
        #y_generated, d_logits_fake, d_params = disc_32(x_generated)
        #y_data, d_logits_real, _ = disc_32(x_data)
        #y_generated, generated_logits, d_params = build_conv_discriminator(x_generated, global_pooling=False)
        #y_data, data_logits, _ = build_conv_discriminator(x_data, global_pooling=False)
        y_data, d_logits_real = discriminator(x_data)
        y_generated, d_logits_fake = discriminator(x_generated, reuse=True)

        eps = 1e-15
        label_smoothing = 0.9
        with tf.variable_scope('discriminator_loss'):
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                        labels=tf.ones_like(y_data) * label_smoothing))
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                        labels=tf.zeros_like(y_generated)))

            d_loss = d_loss_real + d_loss_fake
            #d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
            #d_loss = tf.reduce_mean(d_loss)
        with tf.variable_scope('generator_loss'):
            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                        labels=tf.ones_like(y_generated) * label_smoothing))
            #g_loss = - tf.log(y_generated)
            #g_loss = tf.reduce_mean(g_loss)

        t_vars = tf.trainable_variables()
        d_params = [var for var in t_vars if var.name.startswith('discriminator')]
        g_params = [var for var in t_vars if var.name.startswith('generator')]



        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)

        d_trainer = optimizer.minimize(d_loss, var_list=d_params)
        g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    sess = tf.Session()

    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)

    z_sample_val = np.random.uniform(-1, 1, size=z_shape).astype(np.float32)
    #z_sample_val = np.random.normal(0, 1, size=z_shape).astype(np.float32)

    sum_writer = tf.summary.FileWriter('tb/', sess.graph)
    sum_writer.flush()
    sum_writer.close()

    #iterator = cifar_iterator(batch_size)
    iterator = data_iterator('../datasets/fashion-mnist_train.csv', batch_size)
    #iterator = catdog_iterator(batch_size)
    #iterator = face_iterator(batch_size)
    #iterator = celeb_iterator(batch_size)
    #iterator = mnist_iterator(batch_size)
    #iterator = toy_iterator(batch_size)

    images = iterator.next()
    iterations = images // batch_size
    #bar = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()], maxval=)

    progress_msg = lambda i, j, gen_loss, dsc_loss: \
        "epoch:%s, iter:%s/%s, gen loss:%.5f, dsc loss:%.5f" % (i, j, iterations-1, gen_loss / (j+1), dsc_loss / (j+1))
    sys.stdout.write (progress_msg(0, 0, 0, 0))
    sys.stdout.flush()
    for i in range(sess.run(global_step), max_epoch):
        losses = {'gen': 0, 'dsc': 0}
        for j in range(iterations):
            #x_value, _ = mnist.train.next_batch(batch_size)
            x_value = iterator.next()
            #x_value = 2 * x_value - 1
            z_value = np.random.uniform(-1, 1, size=z_shape).astype(np.float32)
            #z_value = np.random.normal(0, 1, size=z_shape).astype(np.float32)
            _, dsc_loss = sess.run([d_trainer, d_loss],
                    feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            #_, gen_loss = sess.run([g_trainer, g_loss],
            #                       feed_dict={x_data: x_value, z_prior: z_value})
            _, gen_loss = sess.run([g_trainer, g_loss],
                    feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            losses['dsc'] += np.mean(dsc_loss)
            losses['gen'] += np.mean(gen_loss)
            sys.stdout.write ("\r" + progress_msg(i, j, losses['gen'], losses['dsc']))
            sys.stdout.flush()
            if j % 5 == 0:
                x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
                x_gen_val = x_gen_val.transpose(0, 2, 3, 1)
            show_result(x_gen_val)
            key = cv2.waitKey(1)
        sys.stdout.write ('\n')
        sys.stdout.flush()
        #z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        #x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
        #show_result(x_gen_val)
        sess.run(tf.assign(global_step, i + 1))
        if i % 1 == 0:
            saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)


def test():
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    x_generated, _ = generator_64(z_prior)
    chkpt_fname = tf.train.latest_checkpoint(output_path)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, chkpt_fname)

    def randomSeed():
        return np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

    points = 100
    z_points = [randomSeed() for _ in range(points)]
    frames = 100
    i = 0
    start = 0
    end = 1
    while True:
        z_delta = (z_points[end] - z_points[start]) / frames
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_points[start] + z_delta * i})
        show_result(x_gen_val)
        key = cv2.waitKey(16)
        if key == ord('a'):
            i -= 1
            print (i)
        elif key == ord('d'):
            i += 1
            print (i)
        elif key == ord('q'):
            break
        if i >= frames:
            start += 1
            end += 1
        elif i < 0:
            start -= 1
            end -= 1
        i %= frames
        start %= points
        end %= points


if __name__ == '__main__':
    if to_train:
        train()
    else:
        test()
