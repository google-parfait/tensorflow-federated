<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.replace_intrinsics_with_bodies" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.replace_intrinsics_with_bodies

Reduces intrinsics to their bodies as defined in `intrinsic_bodies.py`.

```python
tff.framework.replace_intrinsics_with_bodies(comp)
```

Defined in
[`python/core/impl/intrinsic_reductions.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/intrinsic_reductions.py).

<!-- Placeholder for "Used in" -->

This function operates on the AST level; meaning, it takes in a
`computation_building_blocks.ComputationBuildingBlock` as an argument and
returns one as well. `replace_intrinsics_with_bodies` is intended to be the
standard reduction function, which will reduce all currently implemented
intrinsics to their bodies.

Notice that the success of this function depends on the contract of
`intrinsic_bodies.get_intrinsic_bodies`, that the dict returned by that function
is ordered from more complex intrinsic to less complex intrinsics.

#### Args:

*   <b>`comp`</b>: Instance of
    `computation_building_blocks.ComputationBuildingBlock` in which we wish to
    replace all intrinsics with their bodies.

#### Returns:

An instance of `computation_building_blocks.ComputationBuildingBlock` with all
intrinsics defined in `intrinsic_bodies.py` replaced with their bodies.
