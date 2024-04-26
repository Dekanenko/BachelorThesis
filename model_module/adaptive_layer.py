#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
# you can turn it on if you like; 
# tf.config.experimental.set_visible_devices([], 'GPU')
from tensorflow import keras
import numpy as np


class AdaptiveLayer(keras.layers.Layer):

    """Works as a regular densely-connected NN layer.
    This class inherits keras.layers.Layer.

    `AdaptiveLayer` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is a neural subnetwork, which models an activation.
    `kernel` is a weights matrix created by the layer, and `bias` 
    is a bias vector created by the layer.

    Neural subnetwork is MLP and can contain any number of sublayers and two inner
    activation functions: one for hidden sublayers and the other for an
    output one. From the outside, it works like a regular activation 
    function, outputting values in the same dimensions, see Output shape, 
    yet it is trainable and customizable. One can specify any config they want,
    it will automatically update its output shape if needed.

        Args:
        units: Positive integer, dimensionality of the output space.
        structure: List of positive integers, each element describes the 
        number of units in a sublayer. If the last element != 1, element 1 is
        appended, since each neuron has its own activation function.
        inner_hidden_activation: Activation function to use inside the hidden 
        layers of a subnetwork.
            If you don't specify anything, Tanh activation is applied.
        inner_out_activation: Activation function to use inside the output 
        layer of a subnetwork.
            If you don't specify anything, no outer activation is applied
            (ie. "linear" activation: `a(x) = x`).
        skip_w: Skip connection weight for neural subnetwork. Linear input z 
        is multiplied by this value and added to linear output of a subnetwork, 
        before applying outer activation function:
            `inner_out_activation(a+z*skip_w)`

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units, structure=[2, 2], inner_hidden_activation=tf.nn.tanh,
                 inner_out_activation=lambda x : x, skip_w=0.0):

        if units <= 0:
            raise ValueError("Units must be greater than 0")

        if len(structure) == 0:
            raise ValueError("Structure length must be greater than 0")
        
        for elem in structure:
            if elem <= 0:
                raise ValueError("Structure elements must be greater than 0")
        
        super(AdaptiveLayer, self).__init__()
        self.units = units
        self.structure = structure
        if structure[-1] != 1:
            self.structure.append(1)
        self.inner_hidden_activation = inner_hidden_activation
        self.inner_out_activation = inner_out_activation
        self.skip_w = skip_w
    
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

    """New function. Allows to get the data for illustrating the
    received activation function, which is modeled by subnetwork.

    It skips linear input calculations: `dot(input, kernel) + bias`
    and sends the input values directly to a subnetwork, and returns
    the output.

    Input shape:
        1-D tensor with numbers (x-axis for a function).

    Output shape:
        2-D tensor with shape: `(values (output of a function; y-axis), 
        number of neurons)`.
    """
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


class AdaptiveLayerConv(keras.layers.Layer):

    """Works as a regular densely-connected NN layer.
    This class inherits keras.layers.Layer.

    `AdaptiveLayerConv` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is a neural subnetwork, which models an activation.
    `kernel` is a weights matrix created by the layer, and `bias` 
    is a bias vector created by the layer.

    Neural subnetwork is based on 1D-Convolutions and has two inner
    activation functions: one for hidden sublayers and the other for an
    output one. From the outside, it works like a regular activation 
    function, outputting values in the same dimensions, see Output shape, 
    yet it is trainable and customizable. One can specify any config they want,
    since the output of a subnetwork is automatically defined.

        Args:
        units: Positive integer, dimensionality of the output space.
        structure: List of positive integers, each element describes the 
        number of filters in a sublayer. The last element can have any
        positive value.
        split: the number of `kernels` and `biases` in the outer part of a network.
        For example, having two weight matrices with the same shapes, but different 
        values, allows to obtain two different outputs; this can be utilized by convolutions.
        inner_hidden_activation: Activation function to use inside the hidden 
        layers of a subnetwork.
            If you don't specify anything, Tanh activation is applied
        inner_out_activation: Activation function to use inside the output 
        layer of a subnetwork.
            If you don't specify anything, no outer activation is applied
            (ie. "linear" activation: `a(x) = x`).
        skip_w: Skip connection weight for neural subnetwork. Linear input z 
        is multiplied by this value and added to linear output of a subnetwork, 
        before applying outer activation function:
            `inner_out_activation(a+z*skip_w)`
        noise: Number that Specifies how much noise to insert into a skip connection.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units, structure=[4, 8], split=2, inner_hidden_activation=tf.nn.tanh,
                 inner_out_activation=lambda x : x, skip_w=0.9, noise=0.0):

        if units <= 0:
            raise ValueError("Units must be greater than 0")

        if split <= 0:
            raise ValueError("Split must be greater than 0")

        if len(structure) == 0:
            raise ValueError("Structure length must be greater than 0")
        
        for elem in structure:
            if elem <= 0:
                raise ValueError("Structure elements must be greater than 0")
        
        super(AdaptiveLayerConv, self).__init__()
        self.units = units
        self.structure = structure
        self.split = split
        self.inner_hidden_activation = inner_hidden_activation
        self.inner_out_activation = inner_out_activation
        self.skip_w = skip_w
        self.noise = noise

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

    """New function. Allows to get the data for illustrating the
    received activation function, which is modeled by subnetwork.

    It skips linear input calculations: `dot(input, kernel) + bias`
    and sends the input values directly to a subnetwork, and returns
    the output.

    Input shape:
        1-D tensor with numbers (x-axis for a function).

    Output shape:
        2-D tensor with shape: `(values (output of a function; y-axis), 
        number of neurons)`.
    """
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


class AdaptiveModel(keras.Model):

    """This class inherits keras.Model.
    It is reccomended to use this model only for the purpose of 
    adaptive activation functions study, (ie. their illustration) 
    for AdaptiveLayer and AdaptiveLayerConv.

        Args:
        l: Layers, one should use only AdaptiveLayer and AdaptiveLayerConv.
    """

    def __init__(self, l=[AdaptiveLayer(2), AdaptiveLayer(1)]):
        super().__init__()
        self.l = l

    def call(self, inputs):
        a = inputs
        for layer in self.l:
            a = layer(a)

        return a

    """Allows to get the data about the behavior of adaptive activation
    functions used for each layer (AdaptiveLayer and AdaptiveLayerConv).
    The information obtained can be used for function illustrations.
    
    Input shape:
        1-D tensor with numbers (x-axis for a functions).

    Output shape:
        2-D tensor with shape: `(number of neurons, 
        values - (output of a function; y-axis))`.
    """
    def activation_show(self, inputs):
        y = []

        for layer in self.l:
            tmp = layer.activation_show(inputs)

            for i in range(tmp.shape[-1]):
                y.append(tmp[:, i])

        return y
