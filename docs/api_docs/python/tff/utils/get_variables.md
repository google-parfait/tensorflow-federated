<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.get_variables" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.get_variables

Creates a set of variables that matches the given `type_spec`.

```python
tff.utils.get_variables(
    name,
    type_spec,
    **kwargs
)
```

Defined in
[`python/core/utils/tf_computation_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/tf_computation_utils.py).

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`name`</b>: The common name to use for the scope in which all of the
    variables are to be created.
*   <b>`type_spec`</b>: An instance of
    <a href="../../tff/Type.md"><code>tff.Type</code></a> or something
    convertible to it. The type signature may only be composed of tensor types
    and named tuples, possibly nested.
*   <b>`**kwargs`</b>: Additional keyword args to pass to `tf.get_variable`
    calls.

#### Returns:

Either a single variable when invoked with a tensor TFF type, or a nested
structure of variables created in the appropriately-named variable scopes made
up of anonymous tuples if invoked with a named tuple TFF type.

#### Raises:

*   <b>`TypeError`</b>: if `type_spec` is not a type signature composed of
    tensor and named tuple TFF types.
