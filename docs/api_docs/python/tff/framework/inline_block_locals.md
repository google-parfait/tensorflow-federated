<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.inline_block_locals" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.inline_block_locals

Inlines the block variables in `comp` whitelisted by `variable_names`.

```python
tff.framework.inline_block_locals(
    comp,
    variable_names=None
)
```

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/transformations.py>View
source</a>

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`comp`</b>: The computation building block in which to perform the
    extractions. The names of lambda parameters and block variables in `comp`
    must be unique.
*   <b>`variable_names`</b>: A Python list, tuple, or set representing the
    whitelist of variable names to inline; or None if all variables should be
    inlined.

#### Returns:

A new computation with the transformation applied or the original `comp`.

#### Raises:

*   <b>`ValueError`</b>: If `comp` contains variables with non-unique names.
