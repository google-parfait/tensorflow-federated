<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.build_federated_averaging_process" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.build_federated_averaging_process

```python
tff.learning.build_federated_averaging_process(
    model_fn,
    server_optimizer_fn=(lambda : gradient_descent.SGD(learning_rate=1.0)),
    client_weight_fn=None
)
```

Defined in
[`learning/federated_averaging.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/federated_averaging.py).

<!-- Placeholder for "Used in" -->

Builds the TFF computations for optimization using federated averaging.

#### Args:

*   <b>`model_fn`</b>: A no-arg function that returns a
    <a href="../../tff/learning/TrainableModel.md"><code>tff.learning.TrainableModel</code></a>.
*   <b>`server_optimizer_fn`</b>: A no-arg function that returns a
    `tf.Optimizer`. The `apply_gradients` method of this optimizer is used to
    apply client updates to the server model. The default creates a
    `tf.keras.optimizers.SGD` with a learning rate of 1.0, which simply adds the
    average client delta to the server's model.
*   <b>`client_weight_fn`</b>: Optional function that takes the output of
    `model.report_local_outputs` and returns a tensor that provides the weight
    in the federated average of model deltas. If not provided, the default is
    the total number of examples processed on device.

#### Returns:

A
<a href="../../tff/utils/IterativeProcess.md"><code>tff.utils.IterativeProcess</code></a>.
