"""
Sparse Layer
@author: Lorenzo Buffoni
"""

import tensorflow as tf
import numpy as np


class SparseLayer(tf.keras.layers.Layer):
    def __init__(self, next_layer_dim, activation=None, n_trainable=0):
        super(SparseLayer, self).__init__()

        self.final_shape = next_layer_dim
        self.num_params = n_trainable

        if activation == 'relu':
            self.nonlinear = tf.nn.relu
        elif activation == 'sigmoid':
            self.nonlinear = tf.math.sigmoid
        elif activation == 'tanh':
            self.nonlinear = tf.math.tanh
        elif activation == 'softmax':
            self.nonlinear = tf.nn.softmax
        else:
            self.nonlinear = None

    def build(self, input_shape):
        input_shape = input_shape[1]
        self.dim = input_shape + self.final_shape
        total_params = self.final_shape * input_shape
        self.indexes = np.random.choice(np.array(range(total_params)),
                                        size=self.num_params, replace=False)
        if self.num_params > total_params:
            raise ValueError('Number of trainable parameters exceeds number of total parameters.')

        # construct the variable part
        mask_1 = np.zeros(total_params, dtype=np.float32)
        mask_1[self.indexes] = 1
        self.train_mask = tf.constant(mask_1, dtype=tf.float32)
        limit = np.sqrt(6 / self.dim)
        initializer = np.random.uniform(-limit, limit, total_params)
        self.train_weights = tf.Variable(initializer, trainable=True, dtype=tf.float32)

        # construct the constant part
        mask_2 = np.ones(total_params, dtype=np.float32)
        mask_2[self.indexes] = 0
        self.fixed_mask = tf.constant(mask_2, dtype=tf.float32)
        self.fixed_weights = tf.constant(initializer, dtype=tf.float32)

    def call(self, data, training=False):
        # concatenate the trainable blocks with the constant ones
        self.myweights = tf.math.add(tf.math.multiply(self.train_mask, self.train_weights),
                              tf.math.multiply(self.fixed_mask, self.fixed_weights))
        x = tf.linalg.matmul(data, tf.reshape(self.myweights, [self.dim - self.final_shape, self.final_shape]))
        if self.nonlinear is not None:
            x = self.nonlinear(x)
        return x
