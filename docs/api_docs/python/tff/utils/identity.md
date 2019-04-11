<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.utils.identity" />
<meta itemprop="path" content="Stable" />
</div>

# tff.utils.identity

``` python
tff.utils.identity(source)
```



Defined in [`core/utils/tf_computation_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/core/utils/tf_computation_utils.py).

<!-- Placeholder for "Used in" -->

Applies `tf.identity` pointwise to `source`.

This utility function provides the exact same behavior as `tf.identity`, but
it generalizes to a wider class of objects, including ordinary tensors,
variables, as well as various types of nested structures. It would typically
be used together with `tf.control_dependencies` in non-eager TensorFlow.

#### Args:

* <b>`source`</b>: A nested structure composed of tensors or variables embedded in
    containers that are compatible with `tf.contrib.framework.nest`, or
    instances of `anonymous_tuple.AnonymousTuple`. Elements that represent
    variables have their content extracted prior to identity mapping by first
    invoking `tf.Variable.read_value`.


#### Returns:

The result of applying `tf.identity` to read all elements of the `source`
pointwise, with the same structure as `source`.


#### Raises:

* <b>`TypeError`</b>: If types mismatch.