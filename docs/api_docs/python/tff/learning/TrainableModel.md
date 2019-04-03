<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.TrainableModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="federated_output_computation"/>
<meta itemprop="property" content="input_spec"/>
<meta itemprop="property" content="local_variables"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="forward_pass"/>
<meta itemprop="property" content="report_local_outputs"/>
<meta itemprop="property" content="train_on_batch"/>
</div>

# tff.learning.TrainableModel

## Class `TrainableModel`

Inherits From: [`Model`](../../tff/learning/Model.md)

Defined in
[`learning/model.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/model.py).

<!-- Placeholder for "Used in" -->

A Model with an additional method for (local) training.

This class is primarily intended to be used in the implementation of Federated
Averaging.

## Properties

<h3 id="federated_output_computation"><code>federated_output_computation</code></h3>

Performs federated aggregation of the `Model's` `local_outputs`.

This is typically used to aggregate metrics across many clients, e.g. the body
of the computation might be:

```python
return {
    'num_examples': tff.federated_sum(local_outputs.num_examples),
    'loss': tff.federated_mean(local_outputs.loss)
}
```

N.B. It is assumed all TensorFlow computation happens in the
`report_local_outputs` method, and this method only uses TFF constructs to
specify aggregations across clients.

#### Returns:

Either a <a href="../../tff/Computation.md"><code>tff.Computation</code></a>, or
None if no federated aggregation is needed.

The <a href="../../tff/Computation.md"><code>tff.Computation</code></a> should
take as its single input a
<a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>-placed
<a href="../../tff/Value.md"><code>tff.Value</code></a> corresponding to the
return value of `Model.report_local_outputs`, and return a dictionary or other
structure of <a href="../../tff.md#SERVER"><code>tff.SERVER</code></a>-placed
values; consumers of this method should generally provide these server-placed
values as outputs of the overall computation consuming the model.

<h3 id="input_spec"><code>input_spec</code></h3>

The type specification of the `batch_input` parameter for `forward_pass`.

A nested structure of `tf.TensorSpec` objects, that matches the structure of
arguments that will be passed as the `batch_input` argument of `forward_pass`.
The tensors must include a batch dimension as the first dimension, but the batch
dimension may be undefined.

Similar in spirit to `tf.keras.models.Model.input_spec`.

<h3 id="local_variables"><code>local_variables</code></h3>

An iterable of `tf.Variable` objects, see class comment for details.

<h3 id="non_trainable_variables"><code>non_trainable_variables</code></h3>

An iterable of `tf.Variable` objects, see class comment for details.

<h3 id="trainable_variables"><code>trainable_variables</code></h3>

An iterable of `tf.Variable` objects, see class comment for details.

## Methods

<h3 id="forward_pass"><code>forward_pass</code></h3>

```python
forward_pass(
    batch_input,
    training=True
)
```

Runs the forward pass and returns results.

This method should not modify any variables that are part of the model, that is,
variables that influence the predictions; for that, see
`TrainableModel.train_on_batch`.

However, this method may update aggregated metrics computed across calls to
forward_pass; the final values of such metrics can be accessed via
`aggregated_outputs`.

Uses in TFF:

*   To implement model evaluation.
*   To implement federated gradient descent and other non-Federated-Averaging
    algorithms, where we want the model to run the forward pass and update
    metrics, but there is no optimizer (we might only compute gradients on the
    returned loss).
*   To implement Federated Averaging, when augmented as a `TrainableModel`.

#### Args:

*   <b>`batch_input`</b>: a nested structure that matches the structure of
    `Model.input_spec` and each tensor in `batch_input` satisfies
    `tf.TensorSpec.is_compatible_with()` for the corresponding `tf.TensorSpec`
    in `Model.input_spec`.
*   <b>`training`</b>: If `True`, run the training forward pass, otherwise, run
    in evaluation mode. The semantics are generally the same as the `training`
    argument to `keras.Model.__call__`; this might e.g. influence how dropout or
    batch normalization is handled.

#### Returns:

A `BatchOutput` object. The object must include the `loss` tensor if the model
will be trained via a gradient-based algorithm.

<h3 id="report_local_outputs"><code>report_local_outputs</code></h3>

```python
report_local_outputs()
```

Returns tensors representing values aggregated over `forward_pass` calls.

In federated learning, the values returned by this method will typically be
further aggregated across clients and made available on the server.

This method returns results from aggregating across *all* previous calls to
`forward_pass`, most typically metrics like accuracy and loss. If needed, we may
add a `clear_aggregated_outputs` method, which would likely just run the
initializers on the `local_variables`.

In general, the tensors returned can be an arbitrary function of all the
`tf.Variables` of this model, not just the `local_variables`; for example, this
could return tensors measuring the total L2 norm of the model (which might have
been updated by training).

This method may return arbitrarily shaped tensors, not just scalar metrics. For
example, it could return the average feature vector or a count of how many times
each feature exceed a certain magnitude.

#### Returns:

A structure of tensors (as supported by `tf.contrib.framework.nest`) to be
aggregated across clients.

<h3 id="train_on_batch"><code>train_on_batch</code></h3>

```python
train_on_batch(batch_input)
```

Like `forward_pass`, but updates the model variables.

Typically this will invoke `forward_pass`, with any corresponding side-effects
such as updating metrics.

#### Args:

*   <b>`batch_input`</b>: The current batch, as for `forward_pass`.

#### Returns:

The same `BatchOutput` as `forward_pass`.
