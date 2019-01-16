<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.learning

Defined in
[`learning/__init__.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/__init__.py).

The public API for model developers using federated learning algorithms.

## Modules

[`framework`](../tff/learning/framework.md) module: The public API for contributors who develop federated learning algorithms.

## Classes

[`class Model`](../tff/learning/Model.md): Represents a Model for use in TensorFlow Federated.

## Functions

[`build_federated_averaging_process(...)`](../tff/learning/build_federated_averaging_process.md): Builds the TFF computations for optimization using federated averaging.

[`from_compiled_keras_model(...)`](../tff/learning/from_compiled_keras_model.md):
Builds a <a href="../tff/learning/Model.md"><code>tff.learning.Model</code></a>.

[`from_keras_model(...)`](../tff/learning/from_keras_model.md): Builds a
<a href="../tff/learning/Model.md"><code>tff.learning.Model</code></a>.
