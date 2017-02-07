"""
Like evaluate.py, this script loads back the original training data and extracts the embeddings.
It will also determine the mean embeddings for each of the three input classes and use these
means to transfer styles between the images.
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

    for i in range(Xs.shape[0]):
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
            print('Determining mean representations ...')
            coeff_means = {}
            counts = {}
            while not coord.should_stop():
                type, coeffs = sess.run([type_batch, embedding], feed_dict={phase_train: False})

                for i, (t, c) in enumerate(zip(type, coeffs)):
                    if t not in coeff_means:
                        coeff_means[t] = np.zeros(c.shape)
                        counts[t] = 0
                    coeff_means[t] += c
                    counts[t] += 1

                min_count = np.min(list(counts.values()))
                if len(counts) >= 3 and min_count > 400:
                    for k in coeff_means.keys():
                        coeff_means[k] /= counts[k]
                    break

            # prior knowledge
            video_wim_hurry = 0
            paintings_afremov = 1
            video_disclosure_magnets = 2

            print('Evaluating ...')
            while not coord.should_stop():
                # obtain embeddings and type identifiers
                types, reference, coeffs = sess.run([type_batch, image_batch, embedding],
                                                    feed_dict={phase_train: False})

                # for each coefficient, remove their original mean,
                # then add back a bit of Leonid Afremov
                alpha = 0.25
                beta = 1.0
                for i in range(coeffs.shape[0]):
                    coeffs[i] -= alpha*coeff_means[types[i]]
                    coeffs[i] += beta*coeff_means[paintings_afremov]

                    # simply reversing the coefficients is interesting as well
                    # coeffs[i] = list(reversed(coeffs[i]))

                # ... then fetching the images given the embedding.
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
