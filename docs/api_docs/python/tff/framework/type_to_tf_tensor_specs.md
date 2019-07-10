<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.framework.type_to_tf_tensor_specs" />
<meta itemprop="path" content="Stable" />
</div>

# tff.framework.type_to_tf_tensor_specs

Returns nested structure of `tf.TensorSpec`s for a given TFF type.

```python
tff.framework.type_to_tf_tensor_specs(type_spec)
```

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/impl/type_utils.py>View
source</a>

<!-- Placeholder for "Used in" -->

The dtypes and shapes of the returned `tf.TensorSpec`s match those used by
`tf.data.Dataset`s to indicate the type and shape of their elements. They can be
used, e.g., as arguments in constructing an iterator over a string handle.

#### Args:

*   <b>`type_spec`</b>: Type specification, either an instance of
    `computation_types.Type`, or something convertible to it. Ther type
    specification must be composed of only named tuples and tensors. In all
    named tuples that appear in the type spec, all the elements must be named.

#### Returns:

A nested structure of `tf.TensorSpec`s with the dtypes and shapes of tensors
defined in `type_spec`. The layout of the structure returned is the same as the
layout of the nested type defined by `type_spec`. Named tuples are represented
as dictionaries.
