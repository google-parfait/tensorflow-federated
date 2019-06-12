<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.get_map_of_unbound_references" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.get_map_of_unbound_references

Gets a Python `dict` of the unbound references in `comp`.

```python
tff.framework.get_map_of_unbound_references(comp)
```

Defined in
[`python/core/impl/transformations.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py).

<!-- Placeholder for "Used in" -->

Compuations that are equal will have the same collections of unbounded
references, so it is safe to use `comp` as the key for this `dict` even though a
given compuation may appear in many positions in the AST.

#### Args:

*   <b>`comp`</b>: The computation building block to parse.

#### Returns:

A Python `dict` of elements where keys are the compuations in `comp` and values
are a Python `set` of the names of the unbound references in the subtree of that
compuation.
