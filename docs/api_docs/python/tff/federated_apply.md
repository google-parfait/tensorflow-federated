<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_apply" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_apply

```python
tff.federated_apply(
    fn,
    arg
)
```

Defined in
[`core/api/intrinsics.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py).

<!-- Placeholder for "Used in" -->

Applies a given function to a federated value on the
<a href="../tff.md#SERVER"><code>tff.SERVER</code></a>.

#### Args:

*   <b>`fn`</b>: A function to apply to the member content of `arg` on the
    <a href="../tff.md#SERVER"><code>tff.SERVER</code></a>. The parameter of
    this function must be of the same type as the member constituent of `arg`.
*   <b>`arg`</b>: A value of a TFF federated type placed at the
    <a href="../tff.md#SERVER"><code>tff.SERVER</code></a>, and with the
    `all_equal` bit set.

#### Returns:

A federated value on the <a href="../tff.md#SERVER"><code>tff.SERVER</code></a>
that represents the result of applying `fn` to the member constituent of `arg`.

#### Raises:

*   <b>`TypeError`</b>: If the arguments are not of the appropriate types.
