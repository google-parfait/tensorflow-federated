<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.create_variables" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.create_variables

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/tf_computation_utils.py">View
source</a>

Creates a set of variables that matches the given `type_spec`.

```python
tff.utils.create_variables(
    name,
    type_spec,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

Unlike `tf.get_variables`, this method will always create new variables, and
will not retrieve variables previously created with the same name.

#### Args:

*   <b>`name`</b>: The common name to use for the scope in which all of the
    variables are to be created.
*   <b>`type_spec`</b>: An instance of
    <a href="../../tff/Type.md"><code>tff.Type</code></a> or something
    convertible to it. The type signature may only be composed of tensor types
    and named tuples, possibly nested.
*   <b>`**kwargs`</b>: Additional keyword args to pass to `tf.Variable`
    construction.

#### Returns:

Either a single variable when invoked with a tensor TFF type, or a nested
structure of variables created in the appropriately-named variable scopes made
up of anonymous tuples if invoked with a named tuple TFF type.

#### Raises:

*   <b>`TypeError`</b>: if `type_spec` is not a type signature composed of
    tensor and named tuple TFF types.
