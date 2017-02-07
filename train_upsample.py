"""
Uses the .tfrecord files and the previously trained VAE to train an optimization network
consisting of convolutional, transpose convolutional and residual connections.
This network is used as a post-processing step in an attempt to get some detail back from
the low-res VAE generated image.

It is not cross-validated; training is also extremely slow at the current set-up. I let it
run for approx. 18 hours during which lost about 10% of the training error.
"""

from os import path
from glob import glob

import cv2
import tensorflow as tf
import numpy as np

from libs import import_images
from libs.impex import import_graph
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


def generator(x, name='upsample'):
    with tf.variable_scope(name):
        # forcing the input to have a known name
        x = tf.identity(x, name='x')

        shape = x.get_shape()
        height = shape[1].value
        width = shape[2].value
        output_channels = shape[3].value

        with tf.variable_scope('residuals_lo'):
            with tf.variable_scope('prepass'):
                h, _ = conv2d(x, n_output=32, d_h=1, d_w=1, name='1')
                h = tf.nn.elu(h, name='1/elu')
                x = h

                h, _ = conv2d(h, n_output=32, d_h=2, d_w=2, name='2')
                h = tf.nn.elu(h, name='2/elu')

            channels = [32, 64, 32]
            for i, c in enumerate(channels):
                h, _ = conv2d(h, n_output=c, d_h=1, d_w=1, name='conv/{}'.format(i))
                h = tf.nn.elu(h, name='conv/{}/elu'.format(i))

            h, _ = deconv2d(h,
                            n_output_h=height,
                            n_output_w=width,
                            n_output_ch=32,
                            name='upscaling')
            h = tf.nn.elu(h, name='upscaling/elu')

            h = tf.add(h, x, name='join')

        with tf.variable_scope('clean_pass'):
            h, _ = conv2d(h, n_output=output_channels, d_h=1, d_w=1)
            h = tf.nn.relu(h, name='relu')

        # forcing the output to have a known name
        y = tf.identity(h, name='y')
        return y


def main():
    window = 'preview'
    cv2.namedWindow(window)

    tfrecord_file_names = glob(path.join('data', '*-2.tfrecord.gz'))
    max_reads = 200
    batch_size = 50

    n_epochs = 50
    lr_pretrain = 1e-3
    lr_train = 1e-4
    img_step = 25

    # timestamp = datetime.today().strftime('%Y%m%d-%H%M%S')
    # log_path = path.join('log.upsample', timestamp)
    log_path = path.join('log.upsample', '20170207-021128-2')

    with tf.Graph().as_default() as graph:
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step', dtype=tf.int64)
        learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

        image_batch, type_batch = import_images(tfrecord_file_names, max_reads=max_reads, batch_size=batch_size)

        # forcing the inputs to have a known name
        image_batch = tf.identity(image_batch, name='image_batch')

        # todo: use augmentation; requires tf.map_fn() to work across batches
        # image_batch = tf.image.random_flip_left_right(image_batch)
        # image_batch = tf.image.random_flip_up_down(image_batch)

        import_graph('exported/vae.pb', input_map={'vae/x': image_batch}, prefix='mogrify')
        phase_train = graph.get_tensor_by_name('mogrify/vae/phase_train:0')
        reconstructed = graph.get_tensor_by_name('mogrify/vae/decoder/6/Elu:0')
        reconstructed.set_shape((None, 180, 320, 3))

        # perform simple clipping
        reconstructed = tf.nn.relu(reconstructed, name='mogrify/clip')

        refined = generator(reconstructed, name='refine')

        with tf.variable_scope('training'):
            # programmer is using paranoia. it's super effective.
            # ensure that only the upsampling graph is trained
            vars = [v for v in tf.trainable_variables() if v.name.startswith('refine')]

            # image should be similar to the original ...
            loss_1 = tf.reduce_sum(tf.square(refined - image_batch), axis=3)
            loss_1 = tf.reduce_sum(loss_1, axis=2)
            loss_1 = tf.reduce_sum(loss_1, axis=1)
            loss_1 = tf.reduce_mean(loss_1)

            # ... but dissimilar to the VAE reconstruction ...
            loss_2 = tf.reduce_sum(tf.square(refined - reconstructed), axis=3)
            loss_2 = tf.reduce_sum(loss_2, axis=2)
            loss_2 = tf.reduce_sum(loss_2, axis=1)
            loss_2 = 1.e4/tf.reduce_mean(loss_2)

            # ... and we do both at the same time.
            loss = 0.6*loss_1 + 0.4*loss_2

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
                .minimize(loss, global_step, var_list=vars)

        tf.summary.scalar('loss', loss_1, collections=['test'])
        tf.summary.scalar('learning_rate', learning_rate, collections=['test'])
        tf.summary.image('input', image_batch, collections=['test'])
        tf.summary.image('vae', reconstructed, collections=['test'])
        tf.summary.image('ref', refined, collections=['test'])
        test_summaries = tf.summary.merge_all('test')

    sv = tf.train.Supervisor(graph=graph, logdir=log_path)
    with sv.managed_session() as sess:
        try:
            print('Loading test data ...')

            test_imgs = ('test_1.jpg', 'test_2.jpg', 'test_3.jpg')
            paths = (path.join('test', img_name) for img_name in test_imgs)
            test_images = [cv2.imread(img_path) for img_path in paths]
            test_Xs = np.array(test_images, np.float32) / 255.

            epoch_i = 0
            loss_value = 1e4

            # TODO: needs to be turned off when continuing learning
            batch_i = 0
            while not sv.should_stop() and epoch_i < 1000 and loss_value >= 400:
                # run pre-training on a small subset of the images (here: the visual test images)
                if (batch_i - 1) % img_step == 0:
                    batch_i, loss_value, s, _ = sess.run([global_step, loss, test_summaries, optimizer],
                                                         feed_dict={learning_rate: lr_pretrain,
                                                                    loss_2: 0,
                                                                    image_batch: test_Xs,
                                                                    reconstructed: test_Xs})
                    sv.summary_computed(sess, s, batch_i)
                else:
                    batch_i, loss_value, _ = sess.run([global_step, loss, optimizer],
                                                      feed_dict={learning_rate: lr_pretrain,
                                                                 loss_2: 0,
                                                                 image_batch: test_Xs,
                                                                 reconstructed: test_Xs})
                print('pretrain', batch_i, loss_value)
                epoch_i += 1

                if (batch_i-1) % img_step == 0:
                    # visually evaluate the outcome
                    resized_inputs, results = sess.run([image_batch, refined], feed_dict={image_batch: test_Xs,
                                                                                          reconstructed: test_Xs})
                    assert resized_inputs.shape == results.shape

                    canvas = example_gallery(resized_inputs, results)
                    cv2.imshow(window, canvas)

                # display responsiveness
                if (cv2.waitKey(1) & 0xff) == 27:
                    print('User requested cancellation.')
                    sv.request_stop()
                    break

            epoch_i = 0
            while not sv.should_stop() and epoch_i < n_epochs:
                # run one optimization step
                if (batch_i-1) % img_step == 0:
                    batch_i, loss_value, s, _ = sess.run([global_step, loss, test_summaries, optimizer],
                                                         feed_dict={learning_rate: lr_train,
                                                                    phase_train: False})
                    sv.summary_computed(sess, s, batch_i)
                else:
                    batch_i, loss_value, _ = sess.run([global_step, loss, optimizer],
                                                      feed_dict={learning_rate: lr_train,
                                                                 phase_train: False})
                print('train', batch_i, loss_value)

                if (batch_i-1) % img_step == 0:
                    resized_inputs, results = sess.run([reconstructed,  # reference,
                                                        refined],
                                                       feed_dict={phase_train: False})
                    assert resized_inputs.shape == results.shape

                    resized_inputs = resized_inputs[:3]
                    results = results[:3]

                    canvas = example_gallery(resized_inputs, results)
                    cv2.imshow(window, canvas)

                # display responsiveness
                if (cv2.waitKey(1) & 0xff) == 27:
                    print('User requested cancellation.')
                    sv.request_stop()
                    break

        except tf.errors.OutOfRangeError:
            print('Read all examples.')
        finally:
            sv.request_stop()
            sv.wait_for_stop()

        cv2.destroyWindow(window)


if __name__ == '__main__':
    main()
