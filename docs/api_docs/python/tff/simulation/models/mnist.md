<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.models.mnist" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.simulation.models.mnist

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/models/mnist.py">View
source</a>

An example of an MNIST model function for use with TensorFlow Federated.

<!-- Placeholder for "Used in" -->

## Functions

[`create_keras_model(...)`](../../../tff/simulation/models/mnist/create_keras_model.md):
Returns an instance of `tf.keras.Model` for use with the MNIST example.

[`create_simple_keras_model(...)`](../../../tff/simulation/models/mnist/create_simple_keras_model.md):
Returns an instance of `tf.Keras.Model` with just one dense layer.

[`keras_dataset_from_emnist(...)`](../../../tff/simulation/models/mnist/keras_dataset_from_emnist.md):
Converts `dataset` for use with the output of `create_simple_keras_model`.
