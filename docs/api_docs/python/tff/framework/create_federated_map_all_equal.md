<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.create_federated_map_all_equal" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.create_federated_map_all_equal

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/compiler/building_block_factory.py">View
source</a>

Creates a called federated map of equal values.

```python
tff.framework.create_federated_map_all_equal(
    fn,
    arg
)
```

<!-- Placeholder for "Used in" -->

          Call
         /    \

Intrinsic Tuple | [Comp, Comp]

NOTE: The `fn` is required to be deterministic and therefore should contain no
`building_blocks.CompiledComputations`.

#### Args:

*   <b>`fn`</b>: A `building_blocks.ComputationBuildingBlock` to use as the
    function.
*   <b>`arg`</b>: A `building_blocks.ComputationBuildingBlock` to use as the
    argument.

#### Returns:

A `building_blocks.Call`.

#### Raises:

*   <b>`TypeError`</b>: If any of the types do not match.
