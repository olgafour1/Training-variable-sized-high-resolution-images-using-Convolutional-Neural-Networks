import model_template
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
import keras



class full_conv(model_template.ModelTemplate):
    def __init__(self,n_outputs,dropout_rate,**kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
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

    def spp_layer2(self, input_tensor, levels=[2, 1], name='SPP_layer'):
            '''Multiple Level SPP layer.
               Works for levels=[1, 2, 3, 6].'''

            self.sp_tensor = input_tensor

            with tf.variable_scope(name):
                pool_outputs = []
                for l in levels:
                    pool = gen_nn_ops.max_pool_v2(self.sp_tensor, ksize=[1, tf.math.ceil(
                        tf.math.divide(tf.shape(self.sp_tensor)[1], l)),
                                                                         tf.math.ceil(
                                                                             tf.math.divide(tf.shape(self.sp_tensor)[2],
                                                                                            l)), 1],
                                                  strides=[1, tf.math.floor(
                                                      tf.math.divide(tf.shape(self.sp_tensor)[1], l)),
                                                           tf.math.floor(
                                                               tf.math.divide(tf.shape(self.sp_tensor)[2], l)), 1],
                                                  padding='VALID')

                    pool_outputs.append(tf.reshape(pool, [tf.shape(input_tensor)[0], -1]))

                spp_pool = tf.concat(pool_outputs, 1)

                spp_pool = tf.reshape(spp_pool, (-1, 4 * 256 + 256))
            return spp_pool

    def build(self, input_tensor):

        if input_tensor is None:
            raise ValueError("input_tensor must be a valid tf.Tensor object")
            # input_tensor = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='data')

        self.data = input_tensor
        conv1_1= tf.layers.conv2d(self.data, 32, 5, activation=tf.nn.relu,padding="same",name="conv1_1")

        
        pool1 = tf.layers.max_pooling2d(conv1_1, 4, 2, name="pool1")
        
        conv2_1 = tf.layers.conv2d(pool1, 64, 5, activation=tf.nn.relu, name='conv2_1',padding="same")
        
        pool2 = tf.layers.max_pooling2d(conv2_1, 4, 2, name="pool2")
        

        conv3_1 = tf.layers.conv2d(pool2, 128, 5, activation=tf.nn.relu, name='conv3_1',padding="same")
        
        pool3 = tf.layers.max_pooling2d(conv3_1, 4, 2, name="pool3")

        
        conv4_1 = tf.layers.conv2d(pool3, 256, 5, activation=tf.nn.relu, name='conv4_1',padding="same")
        
        pool4 = tf.layers.max_pooling2d(conv4_1, 4, 2, name="pool4")
        #spp = self.spp_layer2(pool4)

        global_average_pooling = tf.reduce_mean(pool4, axis=[1, 2],name="global_pooling")


        global_dropout = tf.layers.dropout(global_average_pooling, rate=self.dropout_rate, name='global_dropout')

     
        logits = tf.layers.dense(global_dropout, units=self.n_outputs, name="dense_1")



        self.conv1_1    = conv1_1
        self.pool1      = pool1
        self.conv2_1    =conv2_1
        self.pool2      =pool2
        self.conv3_1    =conv3_1
        self.pool3      = pool3
        self.conv4_1    =conv4_1
        self.pool4      =pool4
        #self.spp = spp
        self.global_average_pooing=global_average_pooling
        self.global_dropout=global_dropout
        self.logits=logits
