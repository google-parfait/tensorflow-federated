<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tff.learning

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/__init__.py">View
source</a>

The public API for model developers using federated learning algorithms.

<!-- Placeholder for "Used in" -->

## Modules

[`framework`](../tff/learning/framework.md) module: The public API for
contributors who develop federated learning algorithms.

## Classes

[`class BatchOutput`](../tff/learning/BatchOutput.md): A structure that holds
the output of a
<a href="../tff/learning/Model.md"><code>tff.learning.Model</code></a>.

[`class Model`](../tff/learning/Model.md): Represents a model for use in
TensorFlow Federated.

[`class TrainableModel`](../tff/learning/TrainableModel.md): A Model with an
additional method for (local) training.

## Functions

[`assign_weights_to_keras_model(...)`](../tff/learning/assign_weights_to_keras_model.md):
Assigns a nested structure of TFF weights to a Keras model.

[`build_federated_averaging_process(...)`](../tff/learning/build_federated_averaging_process.md):
Builds the TFF computations for optimization using federated averaging.

[`build_federated_evaluation(...)`](../tff/learning/build_federated_evaluation.md):
Builds the TFF computation for federated evaluation of the given model.

[`build_federated_sgd_process(...)`](../tff/learning/build_federated_sgd_process.md):
Builds the TFF computations for optimization using federated SGD.

[`from_compiled_keras_model(...)`](../tff/learning/from_compiled_keras_model.md):
Builds a <a href="../tff/learning/Model.md"><code>tff.learning.Model</code></a>
for an example mini batch.

[`from_keras_model(...)`](../tff/learning/from_keras_model.md): Builds a
<a href="../tff/learning/Model.md"><code>tff.learning.Model</code></a> for an
example mini batch.

[`state_with_new_model_weights(...)`](../tff/learning/state_with_new_model_weights.md):
Returns a `ServerState` with updated model weights.
