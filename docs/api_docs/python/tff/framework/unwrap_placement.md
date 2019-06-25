<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.unwrap_placement" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.unwrap_placement

Strips `comp`'s placement, returning a single call to map, apply or value.

```python
tff.framework.unwrap_placement(comp)
```

Defined in
[`python/core/impl/transformations.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py).

<!-- Placeholder for "Used in" -->

For this purpose it is necessary to assume that all processing under `comp` is
happening at a single placement.

The other assumptions on inputs of `unwrap_placement` are enumerated as follows:

1.  There is at most one unbound reference under `comp`, which is of federated
    type.
2.  The only intrinsics present here are apply or map, zip, and
    federated_value_at_*.
3.  The type signature of `comp` is federated.
4.  There are no instances of `computation_building_blocks.Data` of federated
    type under `comp`; how these would be handled by a function such as this is
    not entirely clear.

Under these conditions, `unwrap_placement` will produce a single call to
federated_map, federated_apply or federated_value, depending on the placement
and type signature of `comp`. Other than this single map or apply, no intrinsics
will remain under `comp`.

#### Args:

*   <b>`comp`</b>: Instance of
    `computation_building_blocks.ComputationBuildingBlock` satisfying the
    assumptions above.

#### Returns:

A modified version of `comp`, whose root is a single called intrinsic, and
containing no other intrinsics. Equivalent to `comp`.

#### Raises:

*   <b>`TypeError`</b>: If the lone unbound reference under `comp` is not of
    federated type, `comp` itself is not of federated type, or `comp` is not a
    building block.
*   <b>`ValueError`</b>: If we encounter a placement other than the one declared
    by `comp.type_signature`, an intrinsic not present in the whitelist above,
    or `comp` contains more than one unbound reference.
