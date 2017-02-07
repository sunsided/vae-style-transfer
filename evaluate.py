"""
This script loads back the original training data, extracts and randomizes the embeddings and uses
these to generate slightly new images.
"""

from os import path
from glob import glob

import cv2
import tensorflow as tf
import numpy as np

from libs import import_images
from libs.impex import import_graph


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


def main():
    window = 'preview'
    cv2.namedWindow(window)

    tfrecord_file_names = glob(path.join('data', '*-2.tfrecord.gz'))
    max_reads = 200
    batch_size = 50

    with tf.Graph().as_default() as graph:
        image_batch, type_batch = import_images(tfrecord_file_names, max_reads=max_reads, batch_size=batch_size)

        import_graph('exported/vae-refine.pb', input_map={'image_batch': image_batch}, prefix='process')
        phase_train = graph.get_tensor_by_name('process/mogrify/vae/phase_train:0')

        embedding = graph.get_tensor_by_name('process/mogrify/vae/variational/add:0')

        reconstructed = graph.get_tensor_by_name('process/mogrify/clip:0')
        reconstructed.set_shape((None, 180, 320, 3))

        refined = graph.get_tensor_by_name('process/refine/y:0')
        refined.set_shape((None, 180, 320, 3))

    coord = tf.train.Coordinator()
    with tf.Session(graph=graph) as sess:
        init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init)

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            print('Evaluating ...')
            while not coord.should_stop():
                # fetching the embeddings given the inputs ...
                reference, coeffs = sess.run([image_batch, embedding], feed_dict={phase_train: False})

                # ... then salting the embeddings ...
                coeffs += np.random.randn(coeffs.shape[0], coeffs.shape[1])

                # ... then fetching the images given the new embeddings.
                results = sess.run(refined, feed_dict={phase_train: False, embedding: coeffs})

                assert reference.shape == results.shape
                reference = reference[:3]
                results = results[:3]

                canvas = example_gallery(reference, results)
                cv2.imshow(window, canvas)

                if (cv2.waitKey(1000) & 0xff) == 27:
                    print('User requested cancellation.')
                    coord.request_stop()
                    break

        except tf.errors.OutOfRangeError:
            print('Read all examples.')
        finally:
            coord.request_stop()
            coord.join(threads)
            coord.wait_for_stop()

        cv2.destroyWindow(window)


if __name__ == '__main__':
    main()
