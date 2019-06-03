<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.build_federated_sgd_process" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.build_federated_sgd_process

Builds the TFF computations for optimization using federated SGD.

```python
tff.learning.build_federated_sgd_process(
    model_fn,
    server_optimizer_fn=(lambda : tf.keras.optimizers.SGD(learning_rate=0.1)),
    client_weight_fn=None
)
```

Defined in
[`python/learning/federated_sgd.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/federated_sgd.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`model_fn`</b>: A no-arg function that returns a
    <a href="../../tff/learning/TrainableModel.md"><code>tff.learning.TrainableModel</code></a>.
*   <b>`server_optimizer_fn`</b>: A no-arg function that returns a
    `tf.Optimizer`. The `apply_gradients` method of this optimizer is used to
    apply client updates to the server model.
*   <b>`client_weight_fn`</b>: Optional function that takes the output of
    `model.report_local_outputs` and returns a tensor that provides the weight
    in the federated average of model deltas. If not provided, the default is
    the total number of examples processed on device.

#### Returns:

A
<a href="../../tff/utils/IterativeProcess.md"><code>tff.utils.IterativeProcess</code></a>.
