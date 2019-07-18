<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.from_compiled_keras_model" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.from_compiled_keras_model

Builds a
<a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a> for an
example mini batch.

```python
tff.learning.from_compiled_keras_model(
    keras_model,
    dummy_batch
)
```

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/keras_utils.py">View
source</a>

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`keras_model`</b>: A `tf.keras.Model` object that was compiled.
*   <b>`dummy_batch`</b>: A nested structure of values that are convertible to
    *batched* tensors with the same shapes and types as expected by
    `forward_pass()`. The values of the tensors are not important and can be
    filled with any reasonable input value.

#### Returns:

A <a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a>.

#### Raises:

*   <b>`TypeError`</b>: If `keras_model` is not an instance of `tf.keras.Model`.
*   <b>`ValueError`</b>: If `keras_model` was *not* compiled.
