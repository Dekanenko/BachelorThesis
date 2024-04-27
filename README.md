# Layer architecture with embedded adaptive activation functions based on neural subnetworks

### Two layers were created that provide neurons with their own adaptive activation functions. These functions are based on neural subnetworks: the first architecture uses MLP for the subnetwork, whereas the second one – 1D-Convolutions. It allows to create different type of activation functions for the neurons which would adapt themselves to a dataset given.

#

## I Architecture – Adaptive NN

Works as a regular densely-connected NN layer. This class inherits keras.layers.Layer. `AdaptiveLayer` implements the operation: `output = activation(dot(input, kernel) + bias)` where `activation` is a neural subnetwork, which models an activation. `kernel` is a weights matrix created by the layer, and `bias` is a bias vector created by the layer.

![](https://github.com/Dekanenko/BachelorThesis/blob/master/assets/AdaptiveNN.png)

`__init__(self, units, structure, inner_hidden_activation, inner_out_activation, skip_w)`

Neural subnetwork is MLP and can contain any number of sublayers and two inner activation functions: one for hidden sublayers and the other for an output one. From the outside, it works like a regular activation function, outputting values in the same dimensions, see Output shape, yet it is trainable and customizable. One can specify any config they want, it will automatically update its output shape if needed.

Args:
  - units: Positive integer, dimensionality of the output space.
  - structure: List of positive integers, each element describes the number of units in a sublayer. If the last element != 1, element 1 is appended, since each neuron has its own activation function.
  - inner_hidden_activation: Activation function to use inside the hidden layers of a subnetwork. If you don't specify anything, Tanh activation is applied.
  - inner_out_activation: Activation function to use inside the output layer of a subnetwork. If you don't specify anything, no outer activation is applied (ie. "linear" activation: `a(x) = x`).
  - skip_w: Skip connection weight for neural subnetwork. Linear input z is multiplied by this value and added to linear output of a subnetwork, before applying outer activation function: `inner_out_activation(a+z*skip_w)`

Input shape:
  N-D tensor with shape: `(batch_size, ..., input_dim)`. The most common situation would be a 2D input with shape `(batch_size, input_dim)`.

Output shape: N-D tensor with shape: `(batch_size, ..., units)`. For instance, for a 2D input with shape `(batch_size, input_dim)`, the output would have shape `(batch_size, units)`.

#

`activation_show(self, inputs)`

New function. Allows to get the data for illustrating the received activation function, which is modeled by subnetwork. It skips linear input calculations: `dot(input, kernel) + bias` and sends the input values directly to a subnetwork, and returns the output.

Input shape: 1-D tensor with numbers (x-axis for a function).
Output shape: 2-D tensor with shape: `(values (output of a function; y-axis), number of neurons)`.

#

## II Architecture – Adaptive Conv NN

Works as a regular densely-connected NN layer. This class inherits keras.layers.Layer. `AdaptiveLayerConv` implements the operation: `output = activation(dot(input, kernel) + bias)` where `activation` is a neural subnetwork, which models an activation. `kernel` is a weights matrix created by the layer, and `bias`  is a bias vector created by the layer.

![](https://github.com/Dekanenko/BachelorThesis/blob/master/assets/AdaptiveConvNN.png)

`__init__(self, units, structure, split, inner_hidden_activation, inner_out_activation, skip_w, noise)`

Neural subnetwork is based on 1D-Convolutions and has two inner activation functions: one for hidden sublayers and the other for an output one. From the outside, it works like a regular activation function, outputting values in the same dimensions, see Output shape, yet it is trainable and customizable. One can specify any config they want, since the output of a subnetwork is automatically defined.

Args:
  - units: Positive integer, dimensionality of the output space.
  - structure: List of positive integers, each element describes the number of filters in a sublayer. The last element can have any positive value.
  - split: the number of `kernels` and `biases` in the outer part of a network. For example, having two weight matrices with the same shapes, but different values, allows to obtain two different outputs; this can be utilized by convolutions.
  - inner_hidden_activation: Activation function to use inside the hidden layers of a subnetwork. If you don't specify anything, Tanh activation is applied
  - inner_out_activation: Activation function to use inside the output layer of a subnetwork. If you don't specify anything, no outer activation is applied (ie. "linear" activation: `a(x) = x`).
  - skip_w: Skip connection weight for neural subnetwork. Linear input z is multiplied by this value and added to linear output of a subnetwork, before applying outer activation function: `inner_out_activation(a+z*skip_w)`
  - noise: Number that Specifies how much noise to insert into a skip connection.

Input shape: N-D tensor with shape: `(batch_size, ..., input_dim)`. The most common situation would be a 2D input with shape `(batch_size, input_dim)`.

Output shape: N-D tensor with shape: `(batch_size, ..., units)`. For instance, for a 2D input with shape `(batch_size, input_dim)`, the output would have shape `(batch_size, units)`.

# 

`activation_show(self, inputs)` works the same way as for the Adaptive NN.

#

## AdaptiveModel

This class inherits keras.Model. It is reccomended to use this model only for the purpose of adaptive activation functions study, (ie. their illustration) for AdaptiveLayer and AdaptiveLayerConv.

`__init__(self, l)`

Args:
  - l: Layers, one should use only AdaptiveLayer and AdaptiveLayerConv.

#

`activation_show(self, inputs)`

Allows to get the data about the behavior of adaptive activation functions used for each layer (AdaptiveLayer and AdaptiveLayerConv). The information obtained can be used for function illustrations.
    
Input shape: 1-D tensor with numbers (x-axis for a functions).

Output shape: 2-D tensor with shape: `(number of neurons, values - (output of a function; y-axis))

