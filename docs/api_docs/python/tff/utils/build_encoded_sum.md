<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.build_encoded_sum" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.build_encoded_sum

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/encoding_utils.py">View
source</a>

Builds `StatefulAggregateFn` for `values`, to be encoded by `encoders`.

```python
tff.utils.build_encoded_sum(
    values,
    encoders
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`values`</b>: Values to be encoded by the `StatefulAggregateFn`. Must be
    convertible to <a href="../../tff/Value.md"><code>tff.Value</code></a>.
*   <b>`encoders`</b>: A collection of `GatherEncoder` objects to be used for
    encoding `values`. Must have the same structure as `values`.

#### Returns:

A `StatefulAggregateFn` of which `next_fn` encodes the input at
<a href="../../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>, and computes their
sum at <a href="../../tff.md#SERVER"><code>tff.SERVER</code></a>, automatically
splitting the decoding part based on its commutativity with sum.

#### Raises:

*   <b>`ValueError`</b>: If `values` and `encoders` do not have the same
    structure.
*   <b>`TypeError`</b>: If `encoders` are not instances of `GatherEncoder`, or
    if `values` are not compatible with the expected input of the `encoders`.
