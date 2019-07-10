<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.assign" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.assign

Creates an op that assigns `target` from `source`.

```python
tff.utils.assign(
    target,
    source
)
```

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/tf_computation_utils.py>View
source</a>

<!-- Placeholder for "Used in" -->

This utility function provides the exact same behavior as `tf.Variable.assign`,
but it generalizes to a wider class of objects, including ordinary variables as
well as various types of nested structures.

#### Args:

*   <b>`target`</b>: A nested structure composed of variables embedded in
    containers that are compatible with `tf.nest`, or instances of
    `anonymous_tuple.AnonymousTuple`.
*   <b>`source`</b>: A nsested structure composed of tensors, matching that of
    `target`.

#### Returns:

A single op that represents the assignment.

#### Raises:

*   <b>`TypeError`</b>: If types mismatch.
