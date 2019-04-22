<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_broadcast" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_broadcast

Broadcasts a federated value from the
<a href="../tff.md#SERVER"><code>tff.SERVER</code></a> to the
<a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

```python
tff.federated_broadcast(value)
```

Defined in
[`core/api/intrinsics.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`value`</b>: A value of a TFF federated type placed at the
    <a href="../tff.md#SERVER"><code>tff.SERVER</code></a>, all members of which
    are equal (the
    <a href="../tff/FederatedType.md#all_equal"><code>tff.FederatedType.all_equal</code></a>
    property of `value` is `True`).

#### Returns:

A representation of the result of broadcasting: a value of a TFF federated type
placed at the <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>, all
members of which are equal.

#### Raises:

*   <b>`TypeError`</b>: if the argument is not a federated TFF value placed at
    the <a href="../tff.md#SERVER"><code>tff.SERVER</code></a>.
