<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.are_equivalent_types" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.are_equivalent_types

Determines whether `type1` and `type2` are equivalent.

```python
tff.framework.are_equivalent_types(
    type1,
    type2
)
```

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/type_utils.py">View
source</a>

<!-- Placeholder for "Used in" -->

We define equivaence in this context as both types being assignable from
one-another.

#### Args:

*   <b>`type1`</b>: One type.
*   <b>`type2`</b>: Another type.

#### Returns:

`True` iff `type1` anf `type2` are equivalent, or else `False`.
