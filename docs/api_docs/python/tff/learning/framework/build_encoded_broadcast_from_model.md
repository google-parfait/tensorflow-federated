<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.framework.build_encoded_broadcast_from_model" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.framework.build_encoded_broadcast_from_model

Builds `StatefulBroadcastFn` for weights of model returned by `model_fn`.

```python
tff.learning.framework.build_encoded_broadcast_from_model(
    model_fn,
    encoder_fn
)
```

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/framework/encoding_utils.py">View
source</a>

<!-- Placeholder for "Used in" -->

This method creates a `SimpleEncoder` for every weight of model created by
`model_fn`, as returned by `encoder_fn`.

#### Args:

*   <b>`model_fn`</b>: A Python callable with no arguments function that returns
    a
    <a href="../../../tff/learning/Model.md"><code>tff.learning.Model</code></a>.
*   <b>`encoder_fn`</b>: A Python callable with a single argument, which is
    expected to be a `tf.Tensor` of shape and dtype to be encoded. The function
    must return a `tensor_encoding.core.SimpleEncoder`, which expects a
    `tf.Tensor` with compatible type as the input to its `encode` method.

#### Returns:

A `StatefulBroadcastFn` for encoding and broadcasting the weights of model
created by `model_fn`.

#### Raises:

*   <b>`TypeError`</b>: If `model_fn` or `encoder_fn` are not callable objects.
