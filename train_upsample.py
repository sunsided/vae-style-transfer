from os import path
from glob import glob
from datetime import datetime

import cv2
import tensorflow as tf
import numpy as np

from libs import import_images
from libs.utils import conv2d, deconv2d


def normalize_img(img):
    min, max = img.min(), img.max()
    return (img - min) / (max - min)


def example_gallery(Xs, reconstructed):
    if np.isnan(reconstructed).any():
        print('Warning: NaN value detected in reconstruction.')

    slices = []

    for i in range(3):
        org = Xs[i, :, :, :]
        img = reconstructed[i, :, :, :]
        img = normalize_img(img)
        slices.append(np.concatenate((org, img), axis=0))

    return np.concatenate(slices, axis=1)


def upscale_bicubic(x):
    with tf.variable_scope('simple_upscale'):
        shape = x.get_shape()
        size = [2*shape[1].value, 2*shape[2].value]
        return tf.image.resize_bicubic(x, size=size, name='bicubic')


def upscale_bilinear(x):
    with tf.variable_scope('simple_upscale'):
        shape = x.get_shape()
        size = [2*shape[1].value, 2*shape[2].value]
        return tf.image.resize_bilinear(x, size=size, name='bicubic')


def generator(x):
    with tf.variable_scope('upsample'):
        shape = x.get_shape()
        height = shape[1].value
        width = shape[2].value
        channels = shape[3].value

        with tf.variable_scope('residuals_lo'):
            h = x
            for i in range(3):
                h, _ = conv2d(x, n_output=3, d_h=1, d_w=1, name='conv/{}'.format(i))
                h = tf.nn.elu(h)

            h = tf.add(h, x, name='join')
            h_lo = h

        with tf.variable_scope('upsample'):
            h, _ = deconv2d(x,
                            n_output_h=2*height,
                            n_output_w=2*width,
                            n_output_ch=6,
                            name='deconv')
            h = tf.nn.elu(h)

        with tf.variable_scope('residuals_hi'):
            for i in range(2):
                h, _ = conv2d(h, n_output=6, d_h=1, d_w=1, name='conv/{}'.format(i))
                h = tf.nn.elu(h)

            h, _ = conv2d(h, n_output=channels, d_h=1, d_w=1, name='conv/prejoin')
            h = tf.nn.elu(h)

            h_upscaled = upscale_bilinear(h_lo)
            h = tf.add(h, h_upscaled, name='join')

        with tf.variable_scope('clean_pass'):
            h, _ = conv2d(h, n_output=channels, d_h=1, d_w=1)
            h = tf.nn.elu(h)

        return h


def main():
    window = 'preview'
    cv2.namedWindow(window)

    tfrecord_file_names = glob(path.join('data', '*-2.tfrecord.gz'))
    max_reads = 200
    batch_size = 50

    n_epochs = 50
    n_epochs_pretrain = 1000
    keep_prob = 0.8
    n_code = 512
    lr_pretrain = 1e-2
    lr_train = 1e-4
    img_step = 20

    timestamp = datetime.today().strftime('%Y%m%d-%H%M%S')
    log_path = path.join('log.upsample', timestamp)

    with tf.Graph().as_default() as graph:
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step', dtype=tf.int64)
        learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

        image_batch, type_batch = import_images(tfrecord_file_names, max_reads=max_reads, batch_size=batch_size)

        reference = upscale_bicubic(image_batch)
        upsampled = generator(image_batch)

        with tf.variable_scope('training'):
            # todo: covariance loss?
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(upsampled - reference), axis=3))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step)

    sv = tf.train.Supervisor(graph=graph, logdir=log_path)
    with sv.managed_session() as sess:
        # import the VAE metagraph
        vae_saver = tf.train.import_meta_graph('log/vae/model.ckpt-247304.meta')
        vae_saver.restore(sess, 'log/vae/model.ckpt-247304')

        try:
            print('Loading test data ...')

            test_imgs = ('test_1.jpg', 'test_2.jpg', 'test_3.jpg')
            paths = (path.join('test', img_name) for img_name in test_imgs)
            test_images = [cv2.imread(img_path) for img_path in paths]
            test_Xs = np.array(test_images, np.float32) / 255.

            epoch_i = 0
            while not sv.should_stop() and epoch_i < 1000:
                # run pre-training on a small subset of the images (here: the visual test images)
                batch_i, loss_value, _ = sess.run([global_step, loss, optimizer],
                                                  feed_dict={learning_rate: lr_pretrain,
                                                             image_batch: test_Xs})
                print('pretrain', batch_i, loss_value)
                epoch_i += 1

                if batch_i % img_step == 0:
                    # visually evaluate the outcome
                    resized_inputs, results = sess.run([reference, upsampled], feed_dict={image_batch: test_Xs})
                    assert resized_inputs.shape == results.shape

                    canvas = example_gallery(resized_inputs, results)
                    cv2.imshow(window, canvas)

                # display responsiveness
                if (cv2.waitKey(1) & 0xff) == 27:
                    sv.request_stop()
                    break

            epoch_i = 0
            loss_value = 1.
            while not sv.should_stop() and epoch_i < n_epochs and loss_value >= 1e-3:
                # run one optimization step
                batch_i, loss_value, _ = sess.run([global_step, loss, optimizer],
                                                  feed_dict={learning_rate: lr_train})
                print('train', batch_i, loss_value)

                if batch_i % img_step == 0:
                    # visually evaluate the outcome
                    # resized_inputs, results = sess.run([reference, upsampled], feed_dict={image_batch: test_Xs})
                    resized_inputs, results = sess.run([reference, upsampled])
                    assert resized_inputs.shape == results.shape

                    canvas = example_gallery(resized_inputs, results)
                    cv2.imshow(window, canvas)

                # display responsiveness
                if (cv2.waitKey(1) & 0xff) == 27:
                    sv.request_stop()
                    break

        except tf.errors.OutOfRangeError:
            print('Read all examples.')
        finally:
            sv.request_stop()

        cv2.destroyWindow(window)


if __name__ == '__main__':
    main()
