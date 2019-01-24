<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.federated_reduce" />
<meta itemprop="path" content="Stable" />
</div>

# tff.federated_reduce

```python
tff.federated_reduce(
    value,
    zero,
    op
)
```

Defined in
[`core/api/intrinsics.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py).

Reduces `value` from <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a> to
<a href="../tff.md#SERVER"><code>tff.SERVER</code></a> using a reduction `op`.

This method reduces a set of member constituents of a `value` of federated type
`T@CLIENTS` for some `T`, using a given `zero` in the algebra (i.e., the result
of reducing an empty set) of some type `U`, and a reduction operator `op` with
type signature `(<U,T> -> U)` that incorporates a single `T`-typed member
constituent of `value` into the `U`-typed result of partial reduction. In the
special case of `T` equal to `U`, this corresponds to the classical notion of
reduction of a set using a commutative associative binary operator. The
generalized reduction (with `T` not equal to `U`) requires that repeated
application of `op` to reduce a set of `T` always yields the same `U`-typed
result, regardless of the order in which elements of `T` are processed in the
course of the reduction.

#### Args:

*   <b>`value`</b>: A value of a TFF federated type placed at the
    <a href="../tff.md#CLIENTS"><code>tff.CLIENTS</code></a>.
*   <b>`zero`</b>: The result of reducing a value with no constituents.
*   <b>`op`</b>: An operator with type signature `(<U,T> -> U)`, where `T` is
    the type of the constituents of `value` and `U` is the type of `zero` to be
    used in performing the reduction.

#### Returns:

A representation on the <a href="../tff.md#SERVER"><code>tff.SERVER</code></a>
of the result of reducing the set of all member constituents of `value` using
the operator `op` into a single item.

#### Raises:

*   <b>`TypeError`</b>: if the arguments are not of the types specified above.
