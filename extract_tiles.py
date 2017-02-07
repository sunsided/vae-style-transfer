"""
This script is used to extract frames from videos to store them as training data.
"""

from queue import PriorityQueue
from threading import Thread
from random import random
from typing import Tuple
from os import path
from glob import iglob

import cv2
import tensorflow as tf
import numpy as np


def extract_video_frames(queue: PriorityQueue,
                         source: int,
                         cap: cv2.VideoCapture,
                         crop: Tuple[int, int, int, int],
                         target_width: int,
                         target_height: int,
                         frame_step: int=1,
                         display_progress: bool=False):
    window = 'video'
    if display_progress:
        cv2.namedWindow(window)

    while True:
        success, buffer = cap.read()
        if not success:
            break

        # crop borders
        buffer = buffer[crop[0]:-crop[2], crop[1]:-crop[3], :]
        buffer = cv2.resize(buffer, (target_width, target_height), interpolation=cv2.INTER_AREA)

        frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        random_priority = random()
        queue.put((random_priority, (buffer, source)))

        if display_progress:
            cv2.imshow(window, buffer)
            if (cv2.waitKey(33) & 0xff) == 27:
                break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame + frame_step)

    if display_progress:
        cv2.destroyWindow(window)


def load_images(queue: PriorityQueue,
                source: int,
                file_path: str,
                target_width: int,
                target_height: int,
                display_progress: bool=False):
    window = 'image'
    if display_progress:
        cv2.namedWindow(window)

    for file in iglob(path.join(file_path, '**', '*.jpg'), recursive=True):
        buffer = cv2.imread(file)
        buffer = cv2.resize(buffer, (target_width, target_height), interpolation=cv2.INTER_AREA)

        random_priority = random()
        queue.put((random_priority, (buffer, source)))

        if display_progress:
            cv2.imshow(window, buffer)
            if (cv2.waitKey(33) & 0xff) == 27:
                break

    if display_progress:
        cv2.destroyWindow(window)


def main():
    # contains approx. 500 paintings from http://leonidafremov.deviantart.com/gallery/
    image_path = '~/Downloads/Leonid Afremov - Gallery'

    # https://vimeo.com/22328077
    video_file_0 = '~/Downloads/Wim - See You Hurry.mp4'
    video_crop_0 = (64, 32, 64, 32)

    # https://www.youtube.com/watch?v=b_KfnGBtVeA
    video_file_1 = '~/Downloads/Disclosure - Magnets ft. Lorde.mp4'
    video_crop_1 = (150, 0, 150, 1)

    frame_queue = PriorityQueue()

    cap_0 = cv2.VideoCapture(video_file_0)
    cap_1 = cv2.VideoCapture(video_file_1)
    try:
        height, width = cap_0.get(cv2.CAP_PROP_FRAME_HEIGHT), \
                        cap_0.get(cv2.CAP_PROP_FRAME_WIDTH)
        n_frames_0 = cap_0.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_step_0 = 10

        n_frames_1 = cap_1.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_step_1 = 8

        print('Extracting {0:d} and {1:d} frames.'.format(int(n_frames_0 // frame_step_0), int(n_frames_1 // frame_step_1)))

        aspect = float(height) / float(width)
        new_width = 320
        new_height = 180  # int(new_width * aspect)
        assert new_height == 180

        threads = [Thread(target=extract_video_frames,
                          args=(frame_queue, 0, cap_0, video_crop_0, new_width, new_height, frame_step_0, False)),
                   Thread(target=extract_video_frames,
                          args=(frame_queue, 2, cap_1, video_crop_1, new_width, new_height, frame_step_1, False)),
                   Thread(target=load_images,
                          args=(frame_queue, 1, image_path, new_width, new_height, False))]

        print('Loading frames ...')
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        print('Done loading frames.')

    finally:
        cap_0.release()
        cap_1.release()

    print('Writing TFRecords ...')

    display_progress = False
    window = 'preview'
    if display_progress:
        cv2.namedWindow(window)

    writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(path.join('data', 'inputs.tfrecord.gz'), writer_options)

    try:
        while not frame_queue.empty():
            priority, (buffer, source) = frame_queue.get()

            img_raw = buffer.astype(np.uint8).tostring()  # Note OpenCV channel order is BGR

            feature = {
                'type': tf.train.Feature(int64_list=tf.train.Int64List(value=[source])),
                'raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            if display_progress:
                cv2.imshow(window, buffer)
                if (cv2.waitKey(33) & 0xff) == 27:
                    break

            # do something
            frame_queue.task_done()
    finally:
        writer.close()

    if display_progress:
        cv2.destroyWindow(window)


if __name__ == '__main__':
    main()
