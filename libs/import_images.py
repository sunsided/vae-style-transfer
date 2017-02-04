import tensorflow as tf


def import_images(tfrecord_file_names, max_reads=100, batch_size=50):
    with tf.variable_scope('import'):

        training_filename_queue = tf.train.string_input_producer(tfrecord_file_names, num_epochs=None)

        reader_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        reader = tf.TFRecordReader(options=reader_options)

        keys, values = reader.read_up_to(training_filename_queue, max_reads)
        features = tf.parse_example(
            values,
            features={
                'raw': tf.FixedLenFeature([], tf.string),
                'type': tf.FixedLenFeature([], tf.int64)
            })

        types = features['type']
        images = tf.decode_raw(features['raw'], tf.uint8)
        images = tf.reshape(images, shape=(-1, 180, 320, 3))
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)

        image_batch, type_batch = tf.train.shuffle_batch(
            [images, types],
            enqueue_many=True,
            batch_size=batch_size,
            min_after_dequeue=batch_size,
            allow_smaller_final_batch=True,
            capacity=2000,
            name='shuffle_batch')

        return image_batch, type_batch