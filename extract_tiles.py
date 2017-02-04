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
        queue.put((random_priority, (buffer, 0)))

        if display_progress:
            cv2.imshow(window, buffer)
            if (cv2.waitKey(33) & 0xff) == 27:
                break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame + frame_step)

    if display_progress:
        cv2.destroyWindow(window)


def load_images(queue: PriorityQueue,
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
        queue.put((random_priority, (buffer, 1)))

        if display_progress:
            cv2.imshow(window, buffer)
            if (cv2.waitKey(33) & 0xff) == 27:
                break

    if display_progress:
        cv2.destroyWindow(window)


def main():
    image_path = '/home/markus/Downloads/Leonidafremov - Gallery'

    video_file = '/media/markus/Nephthys1/Filme/Musikvideos/Wim - See You Hurry.mp4'
    video_crop = (64, 32, 64, 32)

    frame_queue = PriorityQueue()

    cap = cv2.VideoCapture(video_file)
    try:
        height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), \
                        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_step = 10
        print('Extracting {0:d} frames.'.format(int(n_frames // frame_step)))

        aspect = float(height) / float(width)
        new_width = 320
        new_height = int(new_width * aspect)
        assert new_height == 180

        threads = [Thread(target=extract_video_frames,
                          args=(frame_queue, cap, video_crop, new_width, new_height, frame_step, False)),
                   Thread(target=load_images,
                          args=(frame_queue, image_path, new_width, new_height, False))]

        print('Loading frames ...')
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        print('Done loading frames.')

    finally:
        cap.release()

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
