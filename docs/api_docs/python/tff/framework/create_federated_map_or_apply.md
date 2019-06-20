<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.create_federated_map_or_apply" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.create_federated_map_or_apply

Creates a called federated map or apply depending on `arg`s placement.

```python
tff.framework.create_federated_map_or_apply(
    fn,
    arg
)
```

Defined in
[`python/core/impl/computation_constructing_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_constructing_utils.py).

<!-- Placeholder for "Used in" -->

          Call
         /    \

Intrinsic Tuple | [Comp, Comp]

#### Args:

*   <b>`fn`</b>: A `computation_building_blocks.ComputationBuildingBlock` to use
    as the function.
*   <b>`arg`</b>: A `computation_building_blocks.ComputationBuildingBlock` to
    use as the argument.

#### Returns:

A `computation_building_blocks.Call`.

#### Raises:

*   <b>`TypeError`</b>: If any of the types do not match.
