<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.is_assignable_from" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.is_assignable_from

Determines whether `target_type` is assignable from `source_type`.

```python
tff.framework.is_assignable_from(
    target_type,
    source_type
)
```

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/type_utils.py>View
source</a>

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`target_type`</b>: The expected type (that of the target of the
    assignment).
*   <b>`source_type`</b>: The actual type (that of the source of the
    assignment), tested for being a specialization of the `target_type`.

#### Returns:

`True` iff `target_type` is assignable from `source_type`, or else `False`.

#### Raises:

*   <b>`TypeError`</b>: If the arguments are not TFF types.
