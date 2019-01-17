<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.build_model_delta_optimizer_tff" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.framework.build_model_delta_optimizer_tff

```python
tff.learning.framework.build_model_delta_optimizer_tff(
    model_fn,
    model_to_client_delta_fn,
    server_optimizer_fn=None
)
```

Defined in
[`learning/framework/optimizer_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/framework/optimizer_utils.py).

Constructs complete TFF computations for Federated Averaging or SGD.

This provides the TFF orchestration logic connecting the common server logic
which applies aggregated model deltas to the server model with a ClientDeltaFn
that specifies how weight_deltas are computed on device.

Note: We pass in functions rather than constructed objects so we can ensure any
variables or ops created in constructors are placed in the correct graph.
TODO(b/122081673): This can be simplified once we move fully to TF 2.0.

#### Args:

*   <b>`model_fn`</b>: A no-arg function that returns a
    <a href="../../../tff/learning/Model.md"><code>tff.learning.Model</code></a>.
*   <b>`model_to_client_delta_fn`</b>: A function from a model_fn to a
    `ClientDeltaFn`.
*   <b>`server_optimizer_fn`</b>: A no-arg function that returns a
    `tf.Optimizer`. The apply_gradients method of this optimizer is used to
    apply client updates to the server model. The default returns a
    `tf.train.GradientDescent` with a learning_rate of 1.0, which simply adds
    the average client delta to the server's model.

#### Returns:

A
<a href="../../../tff/utils/IterativeProcess.md"><code>tff.utils.IterativeProcess</code></a>.
