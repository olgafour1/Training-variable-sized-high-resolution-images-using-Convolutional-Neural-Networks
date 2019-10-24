import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import argparse
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split



def _data_path(data_directory:str, name:str) -> str:

    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    return os.path.join(data_directory, f'{name}.tfrecords')


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_labels(mode):
    data = pd.read_csv("medium_1k_dataset.csv")
    if mode=="training":
        training_data = data.loc[data["Split"] == "train"]
        labels_file=data["Medium"].values.tolist()
    elif mode=="validation":
        validation_data=data.loc[data["Split"] == "validation"]
        labels_file = data["Medium"].values.tolist()
    else:
        test_data = data.loc[data["Split"] == "test"]
        labels_file = data["Medium"].values.tolist()


    label_dict = {}
    count = 0
    for label in labels_file:
        if not label in label_dict:
            label_dict[label] = count
            count += 1
    labels = []
    for label in labels_file:
        labels.append(label_dict[label])
    return labels

def return_dataset(image_directory: str, mode: str):
    
    #labels=get_labels()
    data = pd.read_csv("medium_1k_dataset.csv")
    #filenames = data["Image file"].values.tolist()



    # X_train, X_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.10, random_state=42)
    #
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=1)


    if mode == "training":

         training_data = data.loc[data["Split"] == "train"]
         X_train, y_train = training_data["Image file"], get_labels(mode="training")

         filenames, labels = X_train, y_train

    if mode == "validation":
       # filenames, labels = filenames[split_1:split_2], labels[split_1:split_2]
        validation_data = data.loc[data["Split"] == "validation"]
        X_val, y_val = validation_data["Image file"], get_labels(mode="validation")
        filenames, labels = X_val, y_val
    if mode == "test":
      #  filenames, labels = filenames[split_2:], labels[split_2:]
        test_data = data.loc[data["Split"] == "test"]
        X_test, y_test = test_data["Image file"], get_labels(mode="test")

        filenames, labels = X_test, y_test

    filename_pairs = zip(filenames, labels)
    classes_dict=dict()
    for filename, label in filename_pairs:

        if label in classes_dict:

            classes_dict[label].append(filename)
        else:

            classes_dict[label] = [filename]


    return classes_dict



def convert_to(data_set, name: str, data_directory: str, num_shards: int = 1):
    print(f'Processing {name} data')

    num_examples = data_set.shape[0]

    def _process_examples(start_idx: int, end_index: int, filename: str):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index in range(start_idx, end_index):
                sys.stdout.write(f"\rProcessing sample {index+1} of {num_examples}")
                sys.stdout.flush()


                image_raw = data_set[index][0].tostring()


                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(int(data_set[index][2])),
                    'width': _int64_feature(int(data_set[index][3])),
                    #'depth': _int64_feature(data_set[index][0].shape[2]),
                    'image': _bytes_feature(image_raw),
                    'label': _int64_feature(int(data_set[index][1]))
                }))
                writer.write(example.SerializeToString())

    if num_shards == 1:
        _process_examples(0, num_examples, _data_path(data_directory, name))
    else:
        total_examples = num_examples
        samples_per_shard = total_examples // num_shards

        for shard in range(num_shards):
            start_index = shard * samples_per_shard
            end_index = start_index + samples_per_shard
            _process_examples(start_index, end_index, _data_path(data_directory, f'{name}-{shard+1}'))

    print()


def convert_to_tf_record(data_directory: str, image_directory: str):

    def chunks(l, n):

        for i in range(0, len(l), n):

            yield l[i:i + n]


    def tf_records(mode, dict):
        for classname, images in dict.items():


            for index,chunk in enumerate(list(chunks(images,10000))):
                data_set = []
                for image in chunk:

                    image_raw = np.array(Image.open(os.path.join(image_directory,image)))

                    height=image_raw.shape[0]

                    width=image_raw.shape[1]
                    with open(os.path.join(image_directory,image), 'rb') as f:
                        image_byte= f.read()


                    data_set.append((image_byte,int(classname),height,width))


                convert_to(np.array(data_set), "class{}-{}".format((str(classname)),index), os.path.join(data_directory,mode))

    tf_records("validation", return_dataset(image_directory, "validation"))
    tf_records("test", return_dataset(image_directory, "test"))
    tf_records("training",return_dataset(image_directory, "training"))


if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--data-directory',
            default='TFrecords/',
            help='Directory where TFRecords will be stored')
        parser.add_argument(
            '--image-directory',
           default='/gpfs/scratch/bsc28/hpai/storage/data/datasets/raw/met/full',
           help='Directory where Images are stored')

        args = parser.parse_args()
        convert_to_tf_record(args.data_directory,args.image_directory)
