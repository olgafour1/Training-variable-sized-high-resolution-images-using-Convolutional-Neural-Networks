from __future__ import print_function
import tensorflow as tf
import os
import numpy as np


class ModelTemplate:
    """Template for supported models. Supported models are built via the `build()` method.

    The `build()` method must not assume that tensorflow variables/operations
    which were instantiated within `__init__` are live. The initialization code of
    `build()` must be **entirely** self-contained.
    """

    def __init__(self, **kwargs):
        self.weights_path = kwargs.get('weights_path', None)

    def build(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor: tensorflow.Tensor
            A tensor specifying the input to the model

        Returns
        -------
        None
        """

    def restore_weights(self, session):
        """Loads weights from @self.weights_path

        Parameters
        ----------

        session: tf.Session
            Tensorflow session to use for restoring the model weights

        Returns
        -------
        None
        """
        if self.weights_path is None:
            raise ValueError("Weights can not be restored because weights_path has not been provided")
        saver = tf.train.Saver()
        if os.path.isdir(self.weights_path):
            saver.restore(session, tf.train.latest_checkpoint(self.weights_path))
        else:
            saver.restore(session, self.weights_path)