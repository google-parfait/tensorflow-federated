<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_map" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_map

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py">View
source</a>

Maps a federated value on
<a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a> pointwise using a
mapping function.

```python
tff.federated_map(
    mapping_fn,
    value
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`mapping_fn`</b>: A mapping function to apply pointwise to member
    constituents of `value` on each of the participants in
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>. The parameter of
    this function must be of the same type as the member constituents of
    `value`.
*   <b>`value`</b>: A value of a TFF federated type placed at the
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>, or a value that
    can be implicitly converted into a TFF federated type, e.g., by zipping.

#### Returns:

A federated value on <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>
that represents the result of mapping.

#### Raises:

*   <b>`TypeError`</b>: If the arguments are not of the appropriate types.
