<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.create_federated_zip" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.create_federated_zip

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_constructing_utils.py">View
source</a>

Creates a called federated zip.

```python
tff.framework.create_federated_zip(value)
```

<!-- Placeholder for "Used in" -->

          Call
         /    \

Intrinsic Tuple | [Comp, Comp]

This function returns a federated tuple given a `value` with a tuple of
federated values type signature.

#### Args:

*   <b>`value`</b>: A `computation_building_blocks.ComputationBuildingBlock`
    with a `type_signature` of type `computation_types.NamedTupleType`
    containing at least one element.

#### Returns:

A `computation_building_blocks.Call`.

#### Raises:

*   <b>`TypeError`</b>: If any of the types do not match.
*   <b>`ValueError`</b>: If `value` does not contain any elements.
