<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.unique_name_generator" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.unique_name_generator

Yields a new unique name that does not exist in `comp`.

```python
tff.framework.unique_name_generator(
    comp,
    prefix='_var'
)
```

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/computation_constructing_utils.py>View
source</a>

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`comp`</b>: The compuation building block to use as a reference.
*   <b>`prefix`</b>: The prefix to use when generating unique names. If `prefix`
    is `None` or if `comp` contains any name with this prefix, then a unique
    prefix will be generated from random lowercase ascii characters.
