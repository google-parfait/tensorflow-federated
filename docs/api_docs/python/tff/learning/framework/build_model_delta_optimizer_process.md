<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.build_model_delta_optimizer_process" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.framework.build_model_delta_optimizer_process

Constructs
<a href="../../../tff/utils/IterativeProcess.md"><code>tff.utils.IterativeProcess</code></a>
for Federated Averaging or SGD.

```python
tff.learning.framework.build_model_delta_optimizer_process(
    model_fn,
    model_to_client_delta_fn,
    server_optimizer_fn,
    stateful_delta_aggregate_fn=build_stateless_mean(),
    stateful_model_broadcast_fn=build_stateless_broadcaster()
)
```

Defined in
[`python/learning/framework/optimizer_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/framework/optimizer_utils.py).

<!-- Placeholder for "Used in" -->

This provides the TFF orchestration logic connecting the common server logic
which applies aggregated model deltas to the server model with a `ClientDeltaFn`
that specifies how `weight_deltas` are computed on device.

Note: We pass in functions rather than constructed objects so we can ensure any
variables or ops created in constructors are placed in the correct graph.

#### Args:

*   <b>`model_fn`</b>: A no-arg function that returns a
    <a href="../../../tff/learning/Model.md"><code>tff.learning.Model</code></a>.
*   <b>`model_to_client_delta_fn`</b>: A function from a `model_fn` to a
    `ClientDeltaFn`.
*   <b>`server_optimizer_fn`</b>: A no-arg function that returns a
    `tf.Optimizer`. The `apply_gradients` method of this optimizer is used to
    apply client updates to the server model.
*   <b>`stateful_delta_aggregate_fn`</b>: A
    <a href="../../../tff/utils/StatefulAggregateFn.md"><code>tff.utils.StatefulAggregateFn</code></a>
    where the `next_fn` performs a federated aggregation and upates state. That
    is, it has TFF type `(state@SERVER, value@CLIENTS, weights@CLIENTS) ->
    (state@SERVER, aggregate@SERVER)`, where the `value` type is
    <a href="../../../tff/learning/framework/ModelWeights.md#trainable"><code>tff.learning.framework.ModelWeights.trainable</code></a>
    corresponding to the object returned by `model_fn`.
*   <b>`stateful_model_broadcast_fn`</b>: A
    <a href="../../../tff/utils/StatefulBroadcastFn.md"><code>tff.utils.StatefulBroadcastFn</code></a>
    where the `next_fn` performs a federated broadcast and upates state. That
    is, it has TFF type `(state@SERVER, value@SERVER) -> (state@SERVER,
    value@CLIENTS)`, where the `value` type is
    <a href="../../../tff/learning/framework/ModelWeights.md"><code>tff.learning.framework.ModelWeights</code></a>
    corresponding to the object returned by `model_fn`.

#### Returns:

A
<a href="../../../tff/utils/IterativeProcess.md"><code>tff.utils.IterativeProcess</code></a>.
