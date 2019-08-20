<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.sequence_reduce" />
<meta itemprop="path" content="Stable" />
</div>

# tff.sequence_reduce

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/api/intrinsics.py">View
source</a>

Reduces a TFF sequence `value` given a `zero` and reduction operator `op`.

```python
tff.sequence_reduce(
    value,
    zero,
    op
)
```

<!-- Placeholder for "Used in" -->

This method reduces a set of elements of a TFF sequence `value`, using a given
`zero` in the algebra (i.e., the result of reducing an empty sequence) of some
type `U`, and a reduction operator `op` with type signature `(<U,T> -> U)` that
incorporates a single `T`-typed element of `value` into the `U`-typed result of
partial reduction. In the special case of `T` equal to `U`, this corresponds to
the classical notion of reduction of a set using a commutative associative
binary operator. The generalized reduction (with `T` not equal to `U`) requires
that repeated application of `op` to reduce a set of `T` always yields the same
`U`-typed result, regardless of the order in which elements of `T` are processed
in the course of the reduction.

One can also invoke `sequence_reduce` on a federated sequence, in which case the
reductions are performed pointwise; under the hood, we construct an expression
of the form `federated_map(a -> sequence_reduce(x, zero, op), value)`. See also
the discussion on `sequence_map`.

#### Args:

*   <b>`value`</b>: A value that is either a TFF sequence, or a federated
    sequence.
*   <b>`zero`</b>: The result of reducing a sequence with no elements.
*   <b>`op`</b>: An operator with type signature `(<U,T> -> U)`, where `T` is
    the type of the elements of the sequence, and `U` is the type of `zero` to
    be used in performing the reduction.

#### Returns:

The `U`-typed result of reducing elements in the sequence, or if the `value` is
federated, a federated `U` that represents the result of locally reducing each
member constituent of `value`.

#### Raises:

*   <b>`TypeError`</b>: If the arguments are not of the types specified above.
