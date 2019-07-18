<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.replace_selection_from_tuple_with_element" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.replace_selection_from_tuple_with_element

Replaces any selection from a tuple with the underlying tuple element.

```python
tff.framework.replace_selection_from_tuple_with_element(comp)
```

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py">View
source</a>

<!-- Placeholder for "Used in" -->

Replaces any occurences of:

Selection \
Tuple | [Comp, Comp, ...]

with the appropriate Comp, as determined by the `index` or `name` of the
`Selection`.

#### Args:

*   <b>`comp`</b>: The computation building block in which to perform the
    replacements.

#### Returns:

A possibly modified version of comp, without any occurrences of selections from
tuples.

#### Raises:

*   <b>`TypeError`</b>: If `comp` is not an instance of
    `computation_building_blocks.ComputationBuildingBlock`.
