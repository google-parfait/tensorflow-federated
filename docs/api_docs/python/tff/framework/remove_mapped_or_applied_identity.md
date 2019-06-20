<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.remove_mapped_or_applied_identity" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.remove_mapped_or_applied_identity

Removes all the mapped or applied identity functions in `comp`.

```python
tff.framework.remove_mapped_or_applied_identity(comp)
```

Defined in
[`python/core/impl/transformations.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py).

<!-- Placeholder for "Used in" -->

This transform traverses `comp` postorder, matches the following pattern, and
removes all the mapped or applied identity fucntions by replacing the following
computation:

          Call
         /    \

Intrinsic Tuple | [Lambda(x), Comp(y)] \
Ref(x)

Intrinsic(<(x -> x), y>)

with its argument:

Comp(y)

y

#### Args:

*   <b>`comp`</b>: The computation building block in which to perform the
    removals.

#### Returns:

A new computation with the transformation applied or the original `comp`.

#### Raises:

*   <b>`TypeError`</b>: If types do not match.
