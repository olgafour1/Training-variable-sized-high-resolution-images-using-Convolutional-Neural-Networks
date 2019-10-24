import os
from PIL import Image
import numpy as np
import tensorflow as tf
import collections
from flushed_print import print
import pandas as pd
import math
from tensorflow.python.ops import gen_nn_ops

def spp_layer2(input_, levels=[4, 2, 1], name = 'SPP_layer'):
    '''Multiple Level SPP layer.
       Works for levels=[1, 2, 3, 6].'''
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        pool_outputs = []
        for l in levels:
            pool = tf.nn.max_pool(input_, ksize=[1, np.ceil(shape[1] * 1. / l).astype(np.int32), np.ceil(shape[2] * 1. / l).astype(np.int32), 1],
                                      strides=[1, np.floor(shape[1] * 1. / l ).astype(np.int32), np.floor(shape[2] * 1. / l ), 1],
                                      padding='VALID')


            pool_outputs.append(tf.reshape(pool, [shape[0], -1]))

        spp_pool = tf.concat(pool_outputs,1)
    return spp_pool


def build_model(input, n_outputs, is_train):
    input_tensor = tf.placeholder(tf.float32, shape=(1, input[0], input[1], 3), name='input')


    conv1_1 = tf.layers.conv2d(input_tensor, 32, 5, activation=tf.nn.relu, padding="SAME", name="conv1_1")

    pool1 = tf.layers.max_pooling2d(conv1_1, 4, 2, name="pool1")

    conv2_1 = tf.layers.conv2d(pool1, 64, 5, activation=tf.nn.relu, name='conv2_1', padding="SAME")

    pool2 = tf.layers.max_pooling2d(conv2_1, 4, 2, name="pool2")

    conv3_1 = tf.layers.conv2d(pool2, 128, 5, activation=tf.nn.relu, name='conv3_1', padding="SAME")

    pool3 = tf.layers.max_pooling2d(conv3_1, 4, 2, name="pool3")

    conv4_1 = tf.layers.conv2d(pool3, 256, 5, activation=tf.nn.relu, name='conv4_1', padding="SAME")

    #pool4 = tf.layers.max_pooling2d(conv4_1, 4, 2, name="pool4")

    top_layer = tf.reduce_mean(conv4_1, axis=[1, 2], name="global_pooling")
    #top_layer = spp_layer2(conv4_1,name='SPP_layer')

    logits = tf.layers.dense(top_layer, units=n_outputs, name="dense_1")



def calculate_memory(batch_size, layers, number_size):
    def get_trainable_parameters():
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()

            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def feed_forward_pass_memory(layers):
        shapes_mem_count = 0
        for l in layers:


            operators = tf.get_default_graph().get_operation_by_name(l)

            single_layer_mem = 1


            for op in operators.outputs[0].get_shape().as_list():


                if op is None:
                    continue

                single_layer_mem *= op
            shapes_mem_count += single_layer_mem
        return shapes_mem_count

    trainable_count = get_trainable_parameters()


    shapes_mem_count = feed_forward_pass_memory(layers)
    total_memory = number_size * (batch_size * (2 * shapes_mem_count) + (3 * trainable_count))
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)

    return (gbytes)


def get_bucket_sizes(_size_boundaries, _ratio_boundaries, images):
    size_min = np.concatenate(([np.iinfo(np.int32).min], _size_boundaries)).astype(np.float32)

    size_max = np.concatenate((_size_boundaries, [np.iinfo(np.int32).max])).astype(np.float32)

    max_length = max(len(row) for row in _ratio_boundaries)
    ratio_boundaries = np.array(
        [row + [np.finfo(np.float32).max] * (max_length - len(row)) for row in _ratio_boundaries])

    keys = range(sum([len(ratio_boundaries[0]) + 1 for i in range(len(ratio_boundaries) + 1)]))

    groups = {key: [0, 0] for key in keys}

    for idx, row in images.iterrows():

        size = row["Product size"]
        ratio = row["Aspect ratio"]
        size_id = np.where((size >= size_min) & (size < size_max))[0][0]

        if size_id in groups:

            mini_bucket = _ratio_boundaries[size_id - 1]

            ratio_min = np.concatenate(([np.finfo(np.float32).min], mini_bucket)).astype(np.float32)

            ratio_max = np.concatenate((mini_bucket, [np.finfo(np.float32).max])).astype(np.float32)

            ratio_id = np.where((ratio >= ratio_min) & (ratio < ratio_max))[0][0]

            id = ((size_id * (len(_ratio_boundaries[0]) + 1) + ratio_id))

            if (groups[id] == [0, 0]):

                groups[id] = [row["Height"], row["Width"]]


            else:

                if (groups[id][0] < row["Height"]):
                    groups[id][0] = row["Height"]
                if (groups[id][1] < row["Width"]):
                    groups[id][1] = row["Width"]

    groups = collections.OrderedDict(sorted(groups.items()))

    return groups


def set_batch_size(classes, memory_limit, _size_boundaries, _ratio_boundaries, batch_sizes, images):
    indices = []

    for input in [value for (key, value) in get_bucket_sizes(_size_boundaries, _ratio_boundaries, images).items()]:
        if input[0] == 0:
            indices.append(0)
        else:
            memory_list = []
            for batch_size in batch_sizes:
                build_model(input, classes, is_train=True)


                layers = [op.name for op in tf.get_default_graph().get_operations() if
                          op.type in ["Relu", "Softmax", "Placeholder", "Mean", "MaxPool","FusedBatchNorm","MaxPool*","dense_1/BiasAdd"]]

                memory = (calculate_memory(batch_size, layers, 4))
                memory_list.append(memory)

                tf.reset_default_graph()
            try:
                index = memory_list.index(max(filter(lambda x: x < memory_limit, memory_list)))
                indices.append(batch_sizes[index])
            except ValueError:

                indices.append(batch_sizes[0])

    return indices


def get_number_of_images(_size_boundaries, _ratio_boundaries, images):
    size_min = np.concatenate(([np.iinfo(np.int32).min], _size_boundaries)).astype(np.float32)

    size_max = np.concatenate((_size_boundaries, [np.iinfo(np.int32).max])).astype(np.float32)

    max_length = max(len(row) for row in _ratio_boundaries)
    ratio_boundaries = np.array(
        [row + [np.finfo(np.float32).max] * (max_length - len(row)) for row in _ratio_boundaries])

    keys = range(sum([len(ratio_boundaries[0]) + 1 for i in range(len(ratio_boundaries) + 1)]))

    groups = {key: 0 for key in keys}

    for idx, row in images.iterrows():
        size = row["Product size"]
        ratio = row["Aspect ratio"]
        size_id = np.where((size >= size_min) & (size < size_max))[0][0]

        if size_id in groups:
            mini_bucket = _ratio_boundaries[size_id - 1]

            ratio_min = np.concatenate(([np.finfo(np.float32).min], mini_bucket)).astype(np.float32)

            ratio_max = np.concatenate((mini_bucket, [np.finfo(np.float32).max])).astype(np.float32)

            ratio_id = np.where((ratio >= ratio_min) & (ratio < ratio_max))[0][0]

            id = ((size_id * (len(_ratio_boundaries[0]) + 1) + ratio_id))

            if (groups[id] == 0):

                groups[id] = 1

            else:
                groups[id] = groups[id] + 1



    groups = collections.OrderedDict(sorted(groups.items()))

    return groups

def return_number_of_steps(_ratio_boundaries,_size_boundaries,batch_sizes, data, number_of_gpus,memory_limit):



    batch_sizes=set_batch_size(classes=23, memory_limit=memory_limit, _size_boundaries=_size_boundaries,
                                    _ratio_boundaries=_ratio_boundaries,
                                    batch_sizes=batch_sizes, images=data)

    number_of_images=(get_number_of_images(_size_boundaries, _ratio_boundaries, images=data))


    number_of_steps=0
    for i in range(len(number_of_images)):
        try:
            number_of_steps +=math.ceil(number_of_images[i]/batch_sizes[i])

        except ZeroDivisionError:
            pass
    print ("The number of steps is {}".format(number_of_steps))



    return math.floor(number_of_steps/number_of_gpus)

