import matplotlib
matplotlib.use('Agg')
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from read_sizes import set_batch_size
import pandas as pd
import tensorflow.contrib.graph_editor as ge
from get_input import load_config_file
from read_sizes import return_number_of_steps
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument("experiment_name", help="Experiment name.", type=str)
parser.add_argument("--retrain", help="Slurm job ID to resum training for", action='store_true')
parser.add_argument("--lr", help="Learning rate", type=float)
parser.add_argument("--drop_rate", help="Drop rate", type=float)
parser.add_argument("--batch_size", help="Batch size", type=int)
args = parser.parse_args()

parameters = {
    'max_epochs': 1,
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

class_names=[key for (key,value) in label_dict.items()]

image_file = "medium_1k_dataset.csv"
data = images = pd.read_csv(image_file)
training_data = data.loc[data["Split"] == "train"]
memory_limit=14
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
train_n_iter = return_number_of_steps(_ratio_boundaries=training_ratio_boundaries,
                                      _size_boundaries=training_size_boundaries,
                                      batch_sizes=batch_sizes, data=training_data, number_of_gpus=4, memory_limit=memory_limit)-1

val_n_iter = return_number_of_steps(_ratio_boundaries=validation_ratio_boundaries,
                                    _size_boundaries=validation_size_boundaries,
                                    batch_sizes=batch_sizes, data=validation_data, number_of_gpus=4,memory_limit=memory_limit)-1
parameters['steps_to_val'] = int(train_n_iter / 2)


def training_model(input_fn, n_outputs):
    image_batch, labels_batch = input_fn()
    image_batch = tf.cast(image_batch, dtype=tf.float32)

    dropout_rate = tf.placeholder(tf.float32, name="drop_rate")
    # is_train = tf.placeholder(tf.bool,name="is_train")
    with tf.name_scope("model"):
        model = full_conv(n_outputs=n_outputs, dropout_rate=dropout_rate)
        model.build(input_tensor=image_batch)
        logits = model.logits

    predictions = tf.argmax(logits, axis=-1, name="predictions")

    batch_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=tf.one_hot(labels_batch, n_outputs),
        logits=logits)

    # lr_bs = tf.constant(parameters["learning_rate"])
    # bs = tf.cast(tf.shape(image_batch)[0],tf.float32)
    # optimizer = tf.train.AdamOptimizer(lr_bs*bs)
    # global_step = tf.train.get_or_create_global_step()
    # opt_step = optimizer.minimize(batch_loss, global_step=global_step)

    # return batch_loss, predictions, labels_batch, dropout_rate
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
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.

            with tf.device(tf_utils.assign_to_device(id, controller)), tf.name_scope(name):
                # Compute loss and gradients, but don't apply them yet
                loss, preds, labels, dropout_tensor = model_fn(input_fn, n_outputs)
                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    # grads = optimizer.compute_gradients(loss)

                    # g = tf.get_default_graph()
                    # ops = g.get_operations()
                    # for op in ge.filter_ops_from_regex(ops, "^tower_[01]/model/pool1/MaxPool"):
                    #     tf.add_to_collection("checkpoints", op.outputs[0])

                    #grads=gradients(loss, tf.trainable_variables(),checkpoints="memory")
                    grads = tf.gradients(loss, tf.trainable_variables())
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

        _gradients = tf_utils.weighted_average_gradients(tower_grads, weights)

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
model_dir = os.path.abspath(os.path.join(out_path, "model1"))


if not os.path.exists(train_summary_dir):
    os.makedirs(train_summary_dir)
if not os.path.exists(validation_summary_dir):
    os.makedirs(validation_summary_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# if not os.path.exists(image_summary_dir):
#         os.makedirs(image_summary_dir)

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
                # path = saver.save(sess,os.path.join(model_dir, 'model'),
                #           global_step=epoch,
                #           write_meta_graph=False)
                # print(" --> Saved model checkpoint to {}".format(path))

        try:
            predictions_list = []
            true_labels_list = []
            while True:

                # for i_val in range(val_n_iter-1):

                # Validation step
                t0 = time.time()
                feed_dict = {handle: validation_handle, drop_rate: 0.0}
                # feed_dict = {handle: validation_handle}
                metrics, pred, label = sess.run([metrics_op, predictions, labels], feed_dict=feed_dict)

                predictions_list.append(pred)
                true_labels_list.append(label)
                sys.stdout.flush()
                current_step = tf.train.global_step(sess, tf.train.get_global_step())

                # for (image, prediction, la) in zip(images[0], pred, label):
                #     i = 0
                #
                #     if (prediction != la):
                #         #     print(image[0].shape)
                #
                #         fig = plt.figure(figsize=(10, 10))
                #         # define subplot
                #         plt.subplot(1, 2, 1, title=class_names[la])
                #         plt.imshow(image, cmap=plt.get_cmap('gray'))
                #         plt.xticks([])
                #         plt.yticks([])
                #         plt.subplot(1, 2, 2, title=class_names[prediction])
                #         plt.imshow(image, cmap=plt.get_cmap('gray'))
                #         plt.xticks([])
                #         plt.yticks([])
                #         plt.savefig(image_summary_dir + "/" +"{}{}.png".format(class_names[la],i))
                #         i = i + 1


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


