<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.check_intrinsics_whitelisted_for_reduction" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.check_intrinsics_whitelisted_for_reduction

Checks whitelist of intrinsics reducible to aggregate or broadcast.

```python
tff.framework.check_intrinsics_whitelisted_for_reduction(comp)
```

Defined in
[`python/core/impl/transformations.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`comp`</b>: Instance of
    `computation_building_blocks.ComputationBuildingBlock` to check for presence
    of intrinsics not currently immediately reducible to `FEDERATED_AGGREGATE`
    or `FEDERATED_BROADCAST`, or local processing.

#### Raises:

*   <b>`ValueError`</b>: If we encounter an intrinsic under `comp` that is not
    whitelisted as currently reducible.
