<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.build_encoded_broadcast" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.build_encoded_broadcast

Builds `StatefulBroadcastFn` for `values`, to be encoded by `encoders`.

```python
tff.utils.build_encoded_broadcast(
    values,
    encoders
)
```

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/encoding_utils.py">View
source</a>

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`values`</b>: Values to be broadcasted by the `StatefulBroadcastFn`. Must
    be convertible to <a href="../../tff/Value.md"><code>tff.Value</code></a>.
*   <b>`encoders`</b>: A collection of `SimpleEncoder` objects to be used for
    encoding `values`. Must have the same structure as `values`.

#### Returns:

A `StatefulBroadcastFn` of which `next_fn` encodes the input at
<a href="../../tff.md#SERVER"><code>tff.SERVER</code></a>, broadcasts the
encoded representation and decodes the encoded representation at
<a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

#### Raises:

*   <b>`ValueError`</b>: If `values` and `encoders` do not have the same
    structure.
*   <b>`TypeError`</b>: If `encoders` are not instances of `SimpleEncoder`, or
    if `values` are not compatible with the expected input of the `encoders`.
