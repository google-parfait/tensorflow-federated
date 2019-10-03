<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.state_with_new_model_weights" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.state_with_new_model_weights

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/framework/optimizer_utils.py">View
source</a>

Returns a `ServerState` with updated model weights.

```python
tff.learning.state_with_new_model_weights(
    server_state,
    trainable_weights,
    non_trainable_weights
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`server_state`</b>: A server state object returned by an iterative
    training process like
    <a href="../../tff/learning/build_federated_averaging_process.md"><code>tff.learning.build_federated_averaging_process</code></a>.
*   <b>`trainable_weights`</b>: A list of `numpy` arrays in the order of the
    original model's `trainable_variables`.
*   <b>`non_trainable_weights`</b>: A list of `numpy` arrays in the order of the
    original model's `non_trainable_variables`.

#### Returns:

A new server `ServerState` object which can be passed to the `next` method of
the iterative process.
