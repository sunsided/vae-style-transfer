"""
Like evaluate_shift_vectors, this file will take input frames and then apply an embedding shift.
However, this script uses the embeddings to modify video sequences.
"""

from os import path
from glob import glob

import cv2
import tensorflow as tf
import numpy as np

from tqdm import tqdm

from libs import import_images
from libs.impex import import_graph


def main(inputs):
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

        except tf.errors.OutOfRangeError:
            print('Read all examples.')
        finally:
            coord.request_stop()
            coord.join(threads)
            coord.wait_for_stop()

        # prior knowledge
        video_wim_hurry = 0
        paintings_afremov = 1
        video_disclosure_magnets = 2

        for video in inputs:
            name = path.basename(video['file'])
            print('Evaluating {}...'.format(name))

            crop = video['crop']
            target_width, target_height = 320, 180

            cap = cv2.VideoCapture(video['file'])
            height, width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), \
                            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if video['length'] is not None:
                n_frames = int(video['length'] * fps)

            width -= crop[1] + crop[3]
            height -= crop[0] + crop[2]

            writer = cv2.VideoWriter('out-' + name,
                                     fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                     fps=fps,
                                     frameSize=(int(width), int(height)))

            try:
                last_frame = None
                for _ in tqdm(range(n_frames+1)):
                    success, frame = cap.read()
                    if not success:
                        break

                    # prepare for the network
                    buffer = frame[crop[0]:-crop[2], crop[1]:-crop[3], :] / 255.
                    buffer = cv2.resize(buffer, (target_width, target_height), interpolation=cv2.INTER_AREA)

                    # obtain the embeddings for the video frame
                    buffer_ = np.expand_dims(buffer, axis=0)
                    coeffs = sess.run(embedding, feed_dict={image_batch: buffer_, phase_train: False})

                    # adjust the embeddings
                    alpha = 0.25
                    beta = 0.5

                    coeffs -= alpha * coeff_means[video['type']]
                    coeffs += beta * coeff_means[paintings_afremov]

                    # fetching the processed image
                    results = sess.run(refined, feed_dict={phase_train: False, embedding: coeffs})
                    assert results.shape[0] == 1

                    # apply moving average for _some_ temporal smoothing
                    if last_frame is None:
                        last_frame = results
                    else:
                        last_frame = 0.2*last_frame + 0.8*results

                    # prepare the output frame
                    size = (int(width), int(height))
                    video_frame = np.squeeze(last_frame[0])
                    video_frame = cv2.resize(video_frame, size, interpolation=cv2.INTER_LANCZOS4)

                    # clipping some blacks
                    video_frame = (video_frame * 1.05) - 0.05

                    # superimpose the original
                    sw = target_width
                    sh = int(target_width * height / width)
                    buffer = cv2.resize(buffer, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
                    video_frame[10:sh+10, 10:sw+10] = buffer

                    video_frame = np.clip(video_frame * 255., 0., 255.).astype(np.uint8)
                    writer.write(video_frame)

                    # cv2.imshow(window, video_frame)
                    # if (cv2.waitKey(1) & 0xff) == 27:
                    #    print('User requested cancellation.')
                    #    coord.request_stop()
                    #    break

            finally:
                writer.release()
                cap.release()

        cv2.destroyWindow(window)


if __name__ == '__main__':

    inputs = [{'file': '/opt/cadl/Downloads/Wim - See You Hurry.mp4',
               'type': 0,
               'crop': (64, 32, 64, 32),
               'length': None,
               'coeffs': [0.25, 0.5]},
              {'file': '/opt/cadl/Downloads/Disclosure - Magnets ft. Lorde.mp4',
               'type': 2,
               'crop': (150, 0, 150, 1),
               'length': None,
               'coeffs': [0.25, 0.5]},
              {'file': '/opt/cadl/Downloads/Daft Punk - Pentatonix.mp4',
               'type': 0,
               'crop': (0, 0, 1, 1),
               'length': 4 * 60 + 8,
               'coeffs': [0.25, 0.5]}]

    main(inputs)
