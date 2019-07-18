<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_collect" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_collect

Returns a federated value from
<a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a> as a
<a href="../tff.md#SERVER"><code>tff.SERVER</code></a> sequence.

```python
tff.federated_collect(value)
```

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py">View
source</a>

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`value`</b>: A value of a TFF federated type placed at the
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.

#### Returns:

A stream of the same type as the member constituents of `value` placed at the
<a href="../tff.md#SERVER"><code>tff.SERVER</code></a>.

#### Raises:

*   <b>`TypeError`</b>: if the argument is not a federated TFF value placed at
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.
