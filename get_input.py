import numpy as np
import tensorflow as tf
from apply_buckets import bucket_by_ratio_and_size
from flushed_print import print
import glob
import os
import json
from read_sizes import set_batch_size
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib import pyplot

def load_config_file(nfile, abspath=False):

    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)



def read_and_decode(tf_records):
    features = tf.parse_single_example(
        tf_records,

        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64)
        })

    with tf.variable_scope('label'):
        label = tf.cast(features['label'], tf.int32)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

    return features["image"], label, height, width

def _multiply8_padding(image,label):
    out_height=tf.shape(image)[2]
    out_width= tf.shape(image)[3]

    def lcm(x, y):

            greater=tf.cond(x> y, lambda: x, lambda: y)

            # result=tf.cond(tf.not_equal(tf.mod(greater,y),0),lambda: greater+1, lambda:greater)
            # tf.print (result)
            cond=lambda i: tf.not_equal(tf.mod(i,y),0)
            body=lambda i:i+1
            result=tf.while_loop(cond, body, [greater])
            return result

    total_padding_height = lcm(out_height,8)-out_height

    total_padding_width = lcm(out_width, 8) - out_width

    paddings=[[0,0],[0,0],[0,total_padding_height],[0,total_padding_width]]
    image=tf.pad(image,paddings)
    return image, label


def channel4_padding(image, label):
    batch_size=tf.shape(image)[0]
    out_height = tf.shape(image)[1]

    out_width = tf.shape(image)[2]

    channel4=tf.zeros([batch_size, out_height, out_width,1],tf.float16)

    image=tf.concat([image,channel4],axis=3)
    return (image, label)




def input_pipeline(filenames, batch_size,size_boundaries,ratio_boundaries, training):

    n_shards = len(filenames)

    files = tf.data.Dataset.list_files(tf.constant(filenames)).shuffle(n_shards)

    dataset = files.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=4
        )
    )


    dataset = dataset.map(read_and_decode,num_parallel_calls=160)


    if training:
        dataset = dataset.apply(bucket_by_ratio_and_size(batch_size, ratio_boundaries, size_boundaries,training=True))
    else:
        dataset = dataset.apply(bucket_by_ratio_and_size(batch_size, ratio_boundaries, size_boundaries, training=False))



    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset

def get_number_of_images(train_files, n_images=None):
    if not n_images:
        n_images = 0
        for filename in train_files:
            print('Counting images from tfrecord {}'.format(filename))
            n_images += sum(1 for _ in tf.python_io.tf_record_iterator(filename))


    return n_images






