#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
from tensorflow import keras
import numpy as np


# In[13]:


class AdaptiveLayer(keras.layers.Layer):

    def __init__(self, units, structure=[2, 2], inner_hidden_activation=tf.nn.tanh,
                 inner_out_activation=lambda x : x, skip_w=0.0):
        super(AdaptiveLayer, self).__init__()
        self.units = units
        self.structure = structure
        if structure[-1] != 1:
            self.structure.append(1)
        self.inner_hidden_activation = inner_hidden_activation
        self.inner_out_activation = inner_out_activation
        self.skip_w = skip_w

    def get_config(self):
        config = super(AdaptiveLayer, self).get_config()
        config = {
            "units": self.units,
            "structure": self.structure,
            "inner_hidden_activation": keras.saving.serialize_keras_object(self.inner_hidden_activation),
            "inner_out_activation": keras.saving.serialize_keras_object(self.inner_out_activation),
            "skip_w": self.skip_w
        }

        return config

    @classmethod
    def from_config(cls, config):
        units_config = config.pop("units")
        units = keras.saving.deserialize_keras_object(units_config)
        structure_config = config.pop("structure")
        structure = keras.saving.deserialize_keras_object(structure_config)
        skip_w_config = config.pop("skip_w")
        skip_w = keras.saving.deserialize_keras_object(skip_w_config)
        inner_hidden_activation_config = config.pop("inner_hidden_activation")
        inner_hidden_activation = keras.saving.deserialize_keras_object(inner_hidden_activation_config)
        inner_out_activation_config = config.pop("inner_out_activation")
        inner_out_activation = keras.saving.deserialize_keras_object(inner_out_activation_config)

        return cls(units, structure, inner_hidden_activation, inner_out_activation, skip_w, **config)

    def save_own_variables(self, store):
        store["outer_w"] = np.array(self.outer_w)
        store["outer_b"] = np.array(self.outer_b)
        for i in range(0, len(self.structure)):
            store["inner_w_"+str(i)] = np.array(self.inner_w[i])
            store["inner_b_"+str(i)] = np.array(self.inner_b[i])

    def load_own_variables(self, store):
        self.outer_w.assign(store["outer_w"])
        self.outer_b.assign(store["outer_b"])
        for i in range(0, len(self.structure)):
            self.inner_w[i].assign(store["inner_w_"+str(i)])
            self.inner_b[i].assign(store["inner_b_"+str(i)])

    def build(self, input_shape):
        if not hasattr(self, "_build"):
            self._build = True
            self.outer_w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal")
            self.outer_b = self.add_weight(shape=(self.units), initializer="zeros")

            self.inner_w = []
            self.inner_b = []

            for i in range(len(self.structure)):
                if i == 0:
                    self.inner_w.append(self.add_weight(shape=(self.units, 1, self.structure[i]),
                                              initializer=tf.initializers.random_normal(stddev=1.5)))
                else:
                    self.inner_w.append(self.add_weight(shape=(self.units, self.structure[i-1], self.structure[i]),
                                              initializer=tf.initializers.random_normal(stddev=0.9)))

                self.inner_b.append(self.add_weight(shape=(self.units, 1, self.structure[i]), initializer="zeros"))

    def call(self, inputs):
        z = inputs @ self.outer_w + self.outer_b
        a = tf.expand_dims(tf.transpose(z), axis=-1)

        for i in range(len(self.structure)-1):
            a = a @ self.inner_w[i] + self.inner_b[i]
            a = self.inner_hidden_activation(a)

        a = a @ self.inner_w[-1] + self.inner_b[-1]

        a = tf.reshape(tf.transpose(a), shape=(len(inputs), self.units))
        a = self.inner_out_activation(a+z*self.skip_w)

        return a


    def activation_show(self, inputs):
        inner_z = tf.cast(tf.expand_dims(tf.transpose(inputs), axis=-1), dtype=tf.float32)
        a = tf.expand_dims(tf.transpose(inner_z), axis=-1)

        for i in range(len(self.structure)-1):
            a = a @ self.inner_w[i] + self.inner_b[i]
            a = self.inner_hidden_activation(a)

        a = a @ self.inner_w[-1] + self.inner_b[-1]

        a = tf.reshape(tf.transpose(a), shape=(len(inputs), self.units))
        a = self.inner_out_activation(a)

        return a


# In[14]:


class AdaptiveLayerConv(keras.layers.Layer):

    def __init__(self, units, structure=[4, 8], split=2, inner_hidden_activation=tf.nn.tanh,
                 inner_out_activation=lambda x : x, skip_w=0.9, noise=0.0):
        super(AdaptiveLayerConv, self).__init__()
        self.units = units
        self.structure = structure
        self.split = split
        self.inner_hidden_activation = inner_hidden_activation
        self.inner_out_activation = inner_out_activation
        self.skip_w = skip_w
        self.noise = noise

    def get_config(self):
        config = super(AdaptiveLayerConv, self).get_config()
        config = {
            "units": self.units,
            "structure": self.structure,
            "split": self.split,
            "inner_hidden_activation": keras.saving.serialize_keras_object(self.inner_hidden_activation),
            "inner_out_activation": keras.saving.serialize_keras_object(self.inner_out_activation),
            "skip_w": self.skip_w,
            "noise": self.noise
        }

        return config

    @classmethod
    def from_config(cls, config):
        units_config = config.pop("units")
        units = keras.saving.deserialize_keras_object(units_config)
        structure_config = config.pop("structure")
        structure = keras.saving.deserialize_keras_object(structure_config)
        split_config = config.pop("split")
        split = keras.saving.deserialize_keras_object(split_config)
        skip_w_config = config.pop("skip_w")
        skip_w = keras.saving.deserialize_keras_object(skip_w_config)
        noise_config = config.pop("noise")
        noise = keras.saving.deserialize_keras_object(noise_config)
        inner_hidden_activation_config = config.pop("inner_hidden_activation")
        inner_hidden_activation = keras.saving.deserialize_keras_object(inner_hidden_activation_config)
        inner_out_activation_config = config.pop("inner_out_activation")
        inner_out_activation = keras.saving.deserialize_keras_object(inner_out_activation_config)

        return cls(units, structure, split, inner_hidden_activation, inner_out_activation, skip_w, noise, **config)

    def build(self, input_shape):
        self.outer_w = self.add_weight(shape=(self.split, input_shape[-1], self.units),
                                                  initializer="random_normal")
        self.outer_b = self.add_weight(shape=(self.split, self.units), initializer="zeros")

        self.inner_conv = []
        for i in range(0, len(self.structure)):
            self.inner_conv.append(keras.layers.Conv1D(self.structure[i], 1,
                                                        activation=self.inner_hidden_activation))

        self.inner_conv.append(keras.layers.AveragePooling1D(pool_size=self.split, data_format='channels_first'))
        self.inner_conv.append(keras.layers.Conv1D(self.units, 1))

    def call(self, inputs):
        z = []
        for i in range(0, self.split):
            noise = tf.random.truncated_normal(shape=(self.outer_b[i].shape))
            z.append(inputs @ self.outer_w[i] + self.outer_b[i] + noise*self.noise)

        a = tf.cast(z, dtype=tf.float32)
        z = tf.cast(z, dtype=tf.float32)

        for i in range(len(self.inner_conv)-2):
            a = self.inner_conv[i](a)

        a = self.inner_conv[len(self.inner_conv)-2](tf.transpose(a))
        a = self.inner_conv[len(self.inner_conv)-1](tf.transpose(a))

        a = tf.reshape(a, shape=(-1, self.units))
        z_skip = tf.reduce_sum(z, axis=0)/self.split

        a = self.inner_out_activation(a + z_skip*self.skip_w)

        return a

    def activation_show(self, inputs):
        tmp = []
        for i in range(0, self.units):
            tmp.append(inputs)

        tmp = tf.cast(tmp, dtype=tf.float32)
        a = tf.expand_dims(tf.transpose(tmp), axis=0)

        for i in range(len(self.inner_conv)-2):
            a = self.inner_conv[i](a)

        a = self.inner_conv[len(self.inner_conv)-1](a)
        a = tf.reshape(a, shape=(-1, self.units))
        a = self.inner_out_activation(a)

        return a


# In[7]:


class AdaptiveModel(keras.Model):

    def __init__(self, l=[AdaptiveLayer(2), AdaptiveLayer(1)]):
        super().__init__()
        self.l = l

    def call(self, inputs):
        a = inputs
        for layer in self.l:
            a = layer(a)

        return a

    def activation_show(self, inputs):
        y = []

        for layer in self.l:
            tmp = layer.activation_show(inputs)

            for i in range(tmp.shape[-1]):
                y.append(tmp[:, i])

        return y

