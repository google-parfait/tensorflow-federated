<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.transform_type_postorder" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.transform_type_postorder

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/type_utils.py">View
source</a>

Walks type tree of `type_signature` postorder, calling `transform_fn`.

```python
tff.framework.transform_type_postorder(
    type_signature,
    transform_fn
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`type_signature`</b>: Instance of `computation_types.Type` to transform
    recursively.
*   <b>`transform_fn`</b>: Transformation function to apply to each node in the
    type tree of `type_signature`. Must be instance of Python function type.

#### Returns:

A possibly transformed version of `type_signature`, with each node in its tree
the result of applying `transform_fn` to the corresponding node in
`type_signature`.

#### Raises:

*   <b>`TypeError`</b>: If the types don't match the specification above.
