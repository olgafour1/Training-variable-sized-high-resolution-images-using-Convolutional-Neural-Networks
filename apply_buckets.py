import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import tensor_shape
import numpy as np

def bucket_by_ratio_and_size(batch_size, ratio_boundaries, product_size_boundaries, training):
    def ratio_func(_elem, _label, _height, _width):
        height, width = tf.cast(_height, tf.float32), tf.cast(_width, tf.float32)
        ratio = width/height
        return ratio

    def product_size_func(_elem, _label, _height, _width):
        height_size, width_size = tf.cast(_height, tf.int32), tf.cast(_width, tf.int32)
        image_size = height_size * width_size
        return image_size

    def decode(image_bytes, label, height, width):

        with tf.variable_scope('image_decode'):
            image = tf.image.decode_jpeg(image_bytes,channels=3)
            image = tf.cast(image, tf.float32) * (1. / 255)



        return image, label

    def element_to_bucket_id(*args, _ratio_boundaries, _size_boundaries, _ratio_func, _size_func):
        size = tf.cast(_size_func(*args), tf.float32)

        buckets_min = np.concatenate(([np.iinfo(np.int32).min], _size_boundaries)).astype(np.float32)
        buckets_max = np.concatenate((_size_boundaries, [np.iinfo(np.int32).max])).astype(np.float32)
        conditions_c = math_ops.logical_and(
            math_ops.less_equal(buckets_min, size),
            math_ops.less(size, buckets_max))
        size_id = math_ops.reduce_min(array_ops.where(conditions_c))

        ratio = tf.cast(_ratio_func(*args), tf.float32)


        # bucket_key = tf.cast(bucket_id, tf.int64)
        # tf.gather(tf.cast(ratio_boundaries, tf.int64), bucket_id)

        max_length = max(len(row) for row in _ratio_boundaries)
        ratio_boundaries_padded= np.array([row + [np.finfo(np.float32).max] * (max_length - len(row)) for row in _ratio_boundaries])

        offset_id = size_id * (len(ratio_boundaries_padded[0])+ 1)



        buckets_min = tf.concat([[np.finfo(np.float32).min], tf.gather(ratio_boundaries_padded, size_id-1)],axis=0)
        # buckets_max = np.concateate((_ratio_boundaries, [np.iinfo(np.int32).max])).astype(np.float32)
        buckets_min= tf.cast(buckets_min, tf.float32)

        buckets_max = tf.concat([tf.gather(ratio_boundaries_padded,size_id-1), [np.finfo(np.float32).max]], axis=0)
        buckets_max=tf.cast(buckets_max,tf.float32)

        conditions_c = math_ops.logical_and(
            math_ops.less_equal(buckets_min, ratio),
            math_ops.less(ratio, buckets_max))
        ratio_id = math_ops.reduce_min(array_ops.where(conditions_c))





        return offset_id + ratio_id

    def make_padded_shapes(shapes, none_filler=None):
        padded = []
        for shape in nest.flatten(shapes):
            shape = tensor_shape.TensorShape(shape)
            shape = [
                none_filler if d.value is None else d
                for d in shape
            ]
            padded.append(shape)

        return nest.pack_sequence_as(shapes, padded)

    def batching_fn(bucket_id, _batch_size, grouped_dataset,_ratio_boundaries,training):




        grouped_dataset = grouped_dataset.map(decode,num_parallel_calls=160)
        if training:
            grouped_dataset=grouped_dataset.shuffle(200)


        shapes = make_padded_shapes(
            grouped_dataset.output_shapes,
            none_filler=None)

        max_length = max(len(row) for row in _ratio_boundaries)
        ratio_boundaries_padded = np.array(
            [row + [np.finfo(np.float32).max] * (max_length - len(row)) for row in _ratio_boundaries])

        keys = range(sum([len(_ratio_boundaries[0]) + 1 for i in range(len(ratio_boundaries_padded) + 1)]))



        if isinstance(_batch_size, (list,)):
            if len(_batch_size) != len(keys):
                raise ValueError(
                    "len(_batch_sizes) must equal ratio_boundaries) + 1")
            else:

                grouped_dataset = grouped_dataset.padded_batch(tf.gather(tf.cast(batch_size, tf.int64), bucket_id),
                                                               shapes)
        else:
            grouped_dataset = grouped_dataset.padded_batch(batch_size, shapes)

        return grouped_dataset

    def _apply_fn(dataset):
        return dataset.apply(tf.data.experimental.group_by_window(
            lambda elem, label, height, width: element_to_bucket_id(
                elem, label, height, width,
                _ratio_boundaries=ratio_boundaries,
                _size_boundaries=product_size_boundaries,
                _ratio_func=ratio_func,
                _size_func=product_size_func
            ),
            lambda bucket_id, grouped_dataset: batching_fn(bucket_id, batch_size, grouped_dataset, ratio_boundaries,training),
            window_size=5000
        ))

    return _apply_fn
