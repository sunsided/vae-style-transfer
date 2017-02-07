"""
Test script to ensure the .tfrecord files can be read back.
"""

from os import path
from glob import glob

import cv2
import tensorflow as tf

from libs import import_images


def main():
    window = 'preview'
    cv2.namedWindow(window)

    tfrecord_file_names = glob(path.join('data', '*.tfrecord.gz'))
    max_reads = 50
    batch_size = 50

    with tf.Graph().as_default() as graph:
        image_batch, type_batch = import_images(tfrecord_file_names, max_reads=max_reads, batch_size=batch_size)

    coord = tf.train.Coordinator()
    with tf.Session(graph=graph) as sess:
        init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init)

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                Xs = sess.run(image_batch)
                for img in Xs:
                    cv2.imshow(window, img)
                    if (cv2.waitKey(33) & 0xff) == 27:
                        coord.request_stop()
                        break

        except tf.errors.OutOfRangeError:
            print('Read all examples.')
        finally:
            coord.request_stop()
            coord.join(threads)

        cv2.destroyWindow(window)


if __name__ == '__main__':
    main()
