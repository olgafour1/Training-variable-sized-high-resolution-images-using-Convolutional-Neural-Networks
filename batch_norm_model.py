import model_template
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

class full_conv(model_template.ModelTemplate):
    def __init__(self,n_outputs,dropout_rate,is_train,**kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.is_train=is_train
        self.n_outputs = n_outputs
        self.data = None
        self.conv1_1 = None
        self.conv1_1_bn = None
        self.relu1_1=None
        self.pool1 = None
        self.conv2_1 = None
        self.conv2_1_bn = None
        self.relu2_1=None
        self.pool2 = None
        self.conv3_1 = None
        self.conv3_1_bn = None
        self.relu3_1 = None
        self.pool3 = None
        self.conv4_1 = None
        self.conv4_1_bn = None
        self.relu4_1 = None
        self.pool4 = None
        self.global_average_pooling=None
        self.logits = None
        #self.probs = None

    def build(self, input_tensor):
        if input_tensor is None:
            raise ValueError("input_tensor must be a valid tf.Tensor object")
            # input_tensor = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='data')

        self.data = input_tensor
        conv1_1 = tf.layers.conv2d(self.data, 32, 5, activation=None, padding="same", name="conv1_1")
        conv1_1_bn = tf.layers.batch_normalization(conv1_1, training=self.is_train, name="conv1_1_bn")
        relu1_1 = tf.nn.relu(conv1_1_bn, name="relu1_1")
        pool1 = tf.layers.max_pooling2d(relu1_1, 4, 2, name="pool1")

        conv2_1 = tf.layers.conv2d(pool1, 64, 5, activation=None, name='conv2_1', padding="same")

        conv2_1_bn = tf.layers.batch_normalization(conv2_1, training=self.is_train, name="con2_1_bn")

        relu2_1 = tf.nn.relu(conv2_1_bn, name="relu2_1")
        pool2 = tf.layers.max_pooling2d(relu2_1, 4, 2, name="pool2")

        conv3_1 = tf.layers.conv2d(pool2, 128, 5, activation=None, name='conv3_1', padding="same")
        conv3_1_bn = tf.layers.batch_normalization(conv3_1, training=self.is_train, name="conv3_1_bn")

        relu3_1 = tf.nn.relu(conv3_1_bn, name="relu3_1")

        pool3 = tf.layers.max_pooling2d(relu3_1, 4, 2, name="pool3")

        conv4_1 = tf.layers.conv2d(pool3, 256, 5, activation=None, name='conv4_1', padding="same")
        conv4_1_bn = tf.layers.batch_normalization(conv4_1, training=self.is_train, name="conv4_1_bn")
        relu4_1 = tf.nn.relu(conv4_1_bn, name="relu4_1")
        pool4 = tf.layers.max_pooling2d(relu4_1, 4, 2, name="pool4")

        global_average_pooling = tf.reduce_mean(pool4, axis=[1, 2], name="global_pooling")

        global_dropout = tf.layers.dropout(global_average_pooling, rate=self.dropout_rate, name='global_dropout')

        logits = tf.layers.dense(global_dropout, units=self.n_outputs, name="dense_1")

        self.conv1_1 = conv1_1
        self.conv1_1_bn = conv1_1_bn
        self.relu1_1 = relu1_1
        self.pool1 = pool1
        self.conv2_1 = conv2_1
        self.conv2_1_bn = conv2_1_bn
        self.relu2_1 = relu2_1
        self.pool2 = pool2
        self.conv3_1 = conv3_1
        self.conv3_1_bn = conv3_1_bn
        self.relu3_1 = relu3_1
        self.pool3 = pool3
        self.conv4_1 = conv4_1
        self.conv4_1_bn = conv4_1_bn
        self.relu4_1 = relu4_1
        self.pool4 = pool4
        self.global_average_pooing = global_average_pooling
        self.global_dropout = global_dropout
        self.logits = logits

