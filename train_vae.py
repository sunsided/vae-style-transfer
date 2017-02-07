"""
Uses the .tfrecord files to train a variational autoencoder.

The VAE is not cross-validated; the idea here is to make the encoder "get" the original
images as good as possible. I stopped training after approx. 30 hours, after which it was still converging.
"""

from os import path
from glob import glob
from datetime import datetime

import cv2
import tensorflow as tf
import numpy as np

from libs import import_images, VAE


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


def main():
    window = 'preview'
    cv2.namedWindow(window)

    tfrecord_file_names = glob(path.join('data', '*-2.tfrecord.gz'))
    max_reads = 200
    batch_size = 50

    n_epochs = 50
    keep_prob = 0.8
    n_code = 512
    learning_rate = 2e-4
    img_step = 20

    timestamp = datetime.today().strftime('%Y%m%d-%H%M%S')
    log_path = path.join('log', timestamp)

    # TODO: override, continuing from a specific snapshot
    log_path = 'log/20170205-034325-2'

    with tf.Graph().as_default() as graph:
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step', dtype=tf.int64)

        image_batch, type_batch = import_images(tfrecord_file_names, max_reads=max_reads, batch_size=batch_size)

        with tf.variable_scope('vae'):
            ae = VAE(input_shape=(None, 180, 320, 3),
                     convolutional=True,
                     variational=True,
                     n_filters=[64, 64, 64, 128, 128, 192, 256],
                     filter_sizes=[7, 7, 5, 5, 3, 3, 3],
                     n_hidden=None,
                     n_code=n_code,
                     dropout=False,
                     activation=tf.nn.elu)
            loss = ae['cost']

            tf.summary.scalar('loss', loss, collections=['test'])
            tf.summary.image('input', ae['x'], collections=['test'])
            tf.summary.image('reconstructed', ae['y'], collections=['test'])
            test_summaries = tf.summary.merge_all('test')

        with tf.variable_scope('training'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step)

    sv = tf.train.Supervisor(graph=graph, logdir=log_path)
    with sv.managed_session() as sess:
        try:
            batch_i = 0
            epoch_i = 0

            print('Loading test data ...')
            test_Xs = np.array([cv2.imread(path.join('test', 'test_1.jpg')),
                                cv2.imread(path.join('test', 'test_2.jpg')),
                                cv2.imread(path.join('test', 'test_3.jpg'))], np.float32) / 255.

            n_tests = test_Xs.shape[0]
            progress_vid = cv2.VideoWriter('output-{0:s}.mp4'.format(timestamp),
                                           fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                           fps=10.0,
                                           frameSize=(n_tests*320, 2*180))

            while not sv.should_stop() and epoch_i < n_epochs:
                batch_i += 1

                Xs = sess.run(image_batch)
                if batch_i % 10 == 0:
                    train_loss, s, i, _ = sess.run([ae['cost'], test_summaries, global_step, optimizer],
                                                   feed_dict={ae['x']: Xs,
                                                              ae['train']: True,
                                                              ae['keep_prob']: keep_prob})
                    sv.summary_computed(sess, s, global_step=i)
                else:
                    train_loss, _ = sess.run([ae['cost'], optimizer],
                                             feed_dict={ae['x']: Xs,
                                                        ae['train']: True,
                                                        ae['keep_prob']: keep_prob})

                # current batch number and mini-batch training loss
                print(batch_i, train_loss)

                if batch_i % img_step == 0:
                    print('Evaluating at batch {0:d}.'.format(batch_i))
                    reconstructed = sess.run(ae['y'], feed_dict={ae['x']: test_Xs,
                                                                 ae['train']: False,
                                                                 ae['keep_prob']: 1.0})

                    canvas = example_gallery(test_Xs, reconstructed)
                    progress_vid.write((canvas * 255.).astype(np.uint8))

                    cv2.imshow(window, canvas)

                # display responsiveness
                if (cv2.waitKey(1) & 0xff) == 27:
                    sv.request_stop()
                    break

        except tf.errors.OutOfRangeError:
            print('Read all examples.')
        finally:
            progress_vid.release()
            sv.request_stop()

        cv2.destroyWindow(window)


if __name__ == '__main__':
    main()
