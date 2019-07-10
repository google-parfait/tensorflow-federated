<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_value" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_value

Returns a federated value at `placement`, with `value` as the constituent.

```python
tff.federated_value(
    value,
    placement
)
```

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py>View
source</a>

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`value`</b>: A value of a non-federated TFF type to be placed.
*   <b>`placement`</b>: The desired result placement (either
    <a href="../tff.md#SERVER"><code>tff.SERVER</code></a> or
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>).

#### Returns:

A federated value with the given placement `placement`, and the member
constituent `value` equal at all locations.

#### Raises:

*   <b>`TypeError`</b>: If the arguments are not of the appropriate types.
