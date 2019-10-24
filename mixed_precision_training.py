import tensorflow as tf
import glob
import pickle
from model import full_conv
import sys
import os
import time
import tf_utils
import argparse
from flushed_print import print
from get_input import input_pipeline
import numpy as np
from read_sizes import set_batch_size
import pandas as pd
import tensorflow.contrib.graph_editor as ge
from get_input import load_config_file
from read_sizes import return_number_of_steps
from tensorflow.python.ops import gen_nn_ops


parser = argparse.ArgumentParser()
parser.add_argument("experiment_name", help="Experiment name.", type=str)
parser.add_argument("--retrain", help="Slurm job ID to resum training for", action='store_true')
parser.add_argument("--lr", help="Learning rate", type=float)
parser.add_argument("--drop_rate", help="Drop rate", type=float)
parser.add_argument("--batch_size", help="Batch size", type=int)
args = parser.parse_args()

parameters = {
    'max_epochs': 34,
    'learning_rate': 0.001,
    'drop_rate': 0.0,
    'batch_size': 128,
    'seed': 777,
    'steps_to_val': 10000
}

out_dir = '__'.join([args.experiment_name, os.environ['SLURM_JOBID']])
if args.lr:
    parameters['learning_rate'] = args.lr
if args.drop_rate:
    parameters['drop_rate'] = args.drop_rate
if args.batch_size:
    parameters['batch_size'] = args.batch_size

if args.retrain:
    runs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "runs_B2B"))
    candidates = []
    for folder in os.listdir(runs_path):
        if folder.split('__')[0] == args.experiment_name:
            candidates.append(folder)
    if len(candidates) == 0:
        raise ValueError('Impossible to find experiment to resume training')
    if len(candidates) > 1:
        raise ValueError(('More than 1 candidate experiment found to resume training from. '
                          'Please, leave only one of following:\n'
                          ' - {}').format('\n - '.join(candidates)))
    out_dir = candidates[0]

with open("medium_1k_dataset.csv", "rb") as fp:
    labels_file = pd.read_csv(fp)

label_dict = {}
count = 0
for label in labels_file["Medium"]:
    if not label in label_dict:
        label_dict[label] = count
        count += 1
n_labels = len(label_dict)
print("Number of labels: {}".format(n_labels))

image_file = "medium_1k_dataset.csv"
data = images = pd.read_csv(image_file)
training_data = data.loc[data["Split"] == "train"]
memory_limit=13.8
config = load_config_file("config.json")

training_size_boundaries = config["data"]["training"]["size_boundaries"]
training_ratio_boundaries = config["data"]["training"]["ratio_boundaries"]
batch_sizes = config["data"]["training"]["batch_sizes"]

training_sizes = set_batch_size(classes=n_labels, memory_limit=memory_limit, _size_boundaries=training_size_boundaries,
                                _ratio_boundaries=training_ratio_boundaries,
                                batch_sizes=batch_sizes, images=training_data)
validation_data = data.loc[data["Split"] == "validation"]

validation_size_boundaries = config["data"]["validation"]["size_boundaries"]
validation_ratio_boundaries = config["data"]["validation"]["ratio_boundaries"]
batch_sizes = config["data"]["validation"]["batch_sizes"]

validation_sizes = set_batch_size(classes=n_labels, memory_limit=memory_limit, _size_boundaries=validation_size_boundaries,
                                  _ratio_boundaries=validation_ratio_boundaries,
                                  batch_sizes=batch_sizes, images=validation_data)
# train_n_iter = return_number_of_steps(_ratio_boundaries=training_ratio_boundaries,
#                                       _size_boundaries=training_size_boundaries,
#                                       batch_sizes=batch_sizes, data=training_data, number_of_gpus=4, memory_limit=memory_limit)-1
#
# val_n_iter = return_number_of_steps(_ratio_boundaries=validation_ratio_boundaries,
#                                     _size_boundaries=validation_size_boundaries,
#                                     batch_sizes=batch_sizes, data=validation_data, number_of_gpus=4,memory_limit=memory_limit)-1
# parameters['steps_to_val'] = int(train_n_iter / 2)

def build_forward_model(input_tensor):
    conv1_1 = tf.layers.conv2d(input_tensor, 32, 5, activation=tf.nn.relu, padding="same", name="conv1_1")

    pool1 = tf.layers.max_pooling2d(conv1_1, 4, 2, name="pool1")

    conv2_1 = tf.layers.conv2d(pool1, 64, 5, activation=tf.nn.relu, name='conv2_1', padding="same")

    pool2 = tf.layers.max_pooling2d(conv2_1, 4, 2, name="pool2")

    conv3_1 = tf.layers.conv2d(pool2, 128, 5, activation=tf.nn.relu, name='conv3_1', padding="same")

    pool3 = tf.layers.max_pooling2d(conv3_1, 4, 2, name="pool3")

    conv4_1 = tf.layers.conv2d(pool3, 256, 5, activation=tf.nn.relu, name='conv4_1', padding="same")

    pool4 = tf.layers.max_pooling2d(conv4_1, 4, 2, name="pool4")

    top_layer= tf.reduce_mean(pool4, axis=[1,2], name="global_pooling")


    return top_layer



def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
    initializer=None, regularizer=None,
    trainable=True,
    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
    initializer=initializer, regularizer=regularizer,
    trainable=trainable,
    *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable

def training_model(input_fn, n_outputs):
    image_batch, labels_batch = input_fn()
    image_batch = tf.cast(image_batch, dtype=tf.float16)


    dropout_rate = tf.placeholder(tf.float32, name="drop_rate")
    # is_train = tf.placeholder(tf.bool,name="is_train")
    top_layer=build_forward_model(input_tensor=image_batch)
    logits = tf.layers.dense(top_layer, units=n_outputs, name="dense_1")
    logits=tf.cast(logits,tf.float32)

    predictions = tf.argmax(logits, axis=-1, name="predictions")

    batch_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.one_hot(labels_batch, n_outputs),
        logits=logits)


    # lr_bs = tf.constant(parameters["learning_rate"]/128)
    # bs = tf.cast(tf.shape(image_batch)[0],tf.float32)
    # optimizer = tf.train.AdamOptimizer(lr_bs*bs)
    # loss_scale = 512 # Value may need tuning depending on the model
    # gradients, variables = zip(*optimizer.compute_gradients(loss * loss_scale))
    # gradients = [grad / loss_scale for grad in gradients]
    # global_step = tf.train.get_or_create_global_step()
    # #opt_step = optimizer.minimize(batch_loss, global_step=global_step)
    # train_op = optimizer.apply_gradients(zip(gradients, variables),global_step=global_step)

    #return train_op,loss, predictions, labels_batch, dropout_rate
    return batch_loss, predictions, labels_batch, dropout_rate


def create_parallel_optimization(model_fn, input_fn, n_outputs, controller="/cpu:0"):
    # This function is defined below; it returns a list of device ids like
    # `['/gpu:0', '/gpu:1']`
    devices = tf_utils.get_available_gpus()

    # This list keeps track of the gradients per tower and the losses
    tower_grads = []
    losses = []
    preds_list = []
    labels_list = []
    dropout_tensor = None
    weights = []
    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):

            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.

            with tf.device(tf_utils.assign_to_device(id, controller)), tf.variable_scope(
         # Note: This forces trainable variables to be stored as float32
         'fp32_storage', custom_getter=float32_variable_storage_getter):

                # Compute loss and gradients, but don't apply them yet
                loss, preds, labels, dropout_tensor = model_fn(input_fn, n_outputs)

                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    # grads = optimizer.compute_gradients(loss)

                    loss_scale=128

                    grads = tf.gradients(loss*loss_scale, tf.trainable_variables())
                    grads_and_vars = list(zip(grads, tf.trainable_variables()))
                    tower_grads.append(grads_and_vars)

                losses.append(loss)
                preds_list.append(preds)
                labels_list.append(labels)
                weights.append(tf.shape(preds)[0])
                # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()
    with tf.name_scope("apply_gradients"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.

        # This function is defined below; it takes the list of (gradient, variable) lists

        # and turns it into a single (gradient, variables) list.

        _gradients = tf_utils.mp_average_gradients(tower_grads, weights,loss_scale)
        total_bs = tf.reduce_sum(weights)
        optimizer = tf.train.AdamOptimizer((parameters["learning_rate"]/128) * tf.cast(total_bs, tf.float32))
        global_step = tf.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(_gradients, global_step)
        avg_loss = tf.reduce_mean(losses)
        concat_preds = tf.concat(preds_list, 0)
        concat_labels = tf.concat(labels_list, 0)

    return apply_gradient_op, avg_loss, concat_preds, concat_labels, dropout_tensor
    # return apply_gradient_op, avg_loss, concat_preds, concat_labels


def input_fn():
    with tf.device(None):
        # remove any device specifications for the input data
        return iterator.get_next()


dataset_path = "TFrecords/"
train_tfrecords = glob.glob(os.path.join(
    dataset_path,
    "training/*"
))

val_tfrecords = glob.glob(os.path.join(
    dataset_path,
    "validation/*"
))

test_tfrecords = glob.glob(os.path.join(
    dataset_path,
    "test/*"
))

# train_ds = input_pipeline(train_tfrecords,
#                          batch_size=sizes,repetition=True)


train_ds = input_pipeline(train_tfrecords,
                          batch_size=training_sizes,
                          size_boundaries=training_size_boundaries,
                          ratio_boundaries=training_ratio_boundaries,
                          training=True
                          )

# train_n_iter=get_number_of_steps(train_tfrecords,batch_size=training_sizes,ratio_boundaries=training_ratio_boundaries,size_boundaries=training_size_boundaries)

# val_n_iter =get_number_of_steps(val_tfrecords,batch_size=validation_sizes,ratio_boundaries=validation_ratio_boundaries,size_boundaries=validation_size_boundaries)

val_ds = input_pipeline(val_tfrecords,
                        batch_size=validation_sizes,
                        size_boundaries=validation_size_boundaries,
                        ratio_boundaries=validation_ratio_boundaries,
                        training=False
                        )

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, train_ds.output_types, train_ds.output_shapes)

train_iterator = train_ds.make_initializable_iterator()
val_iterator = val_ds.make_initializable_iterator()

# optimizer = tf.train.AdamOptimizer(learning_rate=parameters['learning_rate'])
#train_op,batch_loss, predictions, labels, drop_rate=training_model(input_fn,n_labels)
train_op, batch_loss, predictions, labels, drop_rate = create_parallel_optimization(
    training_model,
    input_fn,
    n_labels
)

acc, acc_op, acc_reset = tf_utils.create_reset_metric(
    tf.metrics.accuracy,
    "stream_accuracy",
    labels=labels,
    predictions=predictions,
    name="train_epoch_accuracy"
)

loss, loss_op, loss_reset = tf_utils.create_reset_metric(
    tf.metrics.mean,
    "stream_loss",
    values=batch_loss,
    name="train_epoch_loss"
)
metrics_op = tf.group([acc_op, loss_op])
metrics_reset = tf.group([acc_reset, loss_reset])

out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "runs_B2B", out_dir))
train_summary_dir = os.path.abspath(os.path.join(out_path, "summaries", "train"))
validation_summary_dir = os.path.abspath(os.path.join(out_path, "summaries", "validation"))
model_dir = os.path.abspath(os.path.join(out_path, "checkpoints"))

if not os.path.exists(train_summary_dir):
    os.makedirs(train_summary_dir)
if not os.path.exists(validation_summary_dir):
    os.makedirs(validation_summary_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

with tf.name_scope("summaries"):
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", acc)
summaries = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph=tf.get_default_graph())
validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, graph=tf.get_default_graph())

session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
saver = tf.train.Saver(max_to_keep=None)
# session_conf.gpu_options.per_process_gpu_memory_fraction = 5
with tf.Session(config=session_conf)  as sess:
    sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

    training_handle = sess.run(train_iterator.string_handle())
    validation_handle = sess.run(val_iterator.string_handle())

    tf.train.export_meta_graph(filename=os.path.join(model_dir, 'CNN-graph.meta'))

    if args.retrain:
        print('Path to restore:', model_dir)
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    for epoch in range(parameters['max_epochs']):
        sess.run(train_iterator.initializer)
        sess.run(val_iterator.initializer)
        # for i_train in range(train_n_iter-1):
        try:
            while True:

                t0 = time.time()
                feed_dict = {handle: training_handle, drop_rate: parameters["drop_rate"]}

                _loss, _, _ = sess.run([batch_loss, train_op, metrics_op], feed_dict=feed_dict)

                current_step = tf.train.global_step(sess, tf.train.get_global_step())
                print("Step: {} --> Loss: {} in {}s".format(current_step,
                                                            _loss,
                                                            time.time() - t0))

                # if current_step % parameters['steps_to_val'] == 0 and current_step != 0:
                # Metrics training
        except   tf.errors.OutOfRangeError:
                train_loss, train_acc, train_summaries = sess.run([loss, acc, summaries])
                train_summary_writer.add_summary(train_summaries, current_step)
                print(" --> Step {} --> Training loss: {}, Training acc: {}".format(
                    current_step,
                    train_loss,
                    train_acc)
                )

                # Reset streaming metrics

                sess.run(metrics_reset)
                path = saver.save(sess, os.path.join(model_dir, 'model0'),
                          global_step=epoch,
                          write_meta_graph=False)
                print(" --> Saved model checkpoint to {}".format(path))

        try:
            while True:

                # for i_val in range(val_n_iter-1):

                # Validation step
                t0 = time.time()
                feed_dict = {handle: validation_handle, drop_rate: 0.0}
                # feed_dict = {handle: validation_handle}
                metrics, pred, label = sess.run([metrics_op, predictions, labels], feed_dict=feed_dict)

                sys.stdout.flush()
                current_step = tf.train.global_step(sess, tf.train.get_global_step())

        except tf.errors.OutOfRangeError:
                val_loss, val_acc, val_summaries = sess.run([loss, acc, summaries])
                validation_summary_writer.add_summary(val_summaries, current_step)

                print(" --> Step {} --> Validation loss: {}, Validation acc: {}".format(
                    current_step,
                    val_loss,
                    val_acc)
                )
                # Metrics validation epoch

                # Reset streaming metrics
                sess.run(metrics_reset)
