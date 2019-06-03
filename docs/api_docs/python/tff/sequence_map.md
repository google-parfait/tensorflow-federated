<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.sequence_map" />
<meta itemprop="path" content="Stable" />
</div>

# tff.sequence_map

Maps a TFF sequence `value` pointwise using a given function `mapping_fn`.

```python
tff.sequence_map(
    mapping_fn,
    value
)
```

Defined in
[`python/core/api/intrinsics.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py).

<!-- Placeholder for "Used in" -->

This function supports two modes of usage:

*   When applied to a non-federated sequence, it maps individual elements of the
    sequence pointwise. If the supplied `mapping_fn` is of type `T->U` and the
    sequence `value` is of type `T*` (a seqeunce of `T`-typed elements), the
    result is a sequence of type `U*` (a sequence of `U`-typed elements), with
    each element of the input sequence individually mapped by `mapping_fn`. In
    this mode of usage, `sequence_map` behaves like a compuatation with type
    signature `<T->U,T*> -> U*`.

*   When applied to a federated sequence, `sequence_map` behaves as if it were
    individually applied to each member constituent. In this mode of usage, one
    can think of `sequence_map` as a specialized variant of `federated_map` or
    `federated_apply` that is designed to work with sequences and allows one to
    specify a `mapping_fn` that operates at the level of individual elements.
    Indeed, under the hood, when `sequence_map` is invoked on a federated type,
    it injects one of the `federated_map` or `federated_apply` variants, thus
    emitting expressions like `federated_map(a -> sequence_map(mapping_fn, x),
    value)`.

#### Args:

*   <b>`mapping_fn`</b>: A mapping function to apply pointwise to elements of
    `value`.
*   <b>`value`</b>: A value of a TFF type that is either a sequence, or a
    federated sequence.

#### Returns:

A sequence with the result of applying `mapping_fn` pointwise to each element of
`value`, or if `value` was federated, a federated sequence with the result of
invoking `sequence_map` on member sequences locally and independently at each
location.

#### Raises:

*   <b>`TypeError`</b>: If the arguments are not of the appropriate types.
