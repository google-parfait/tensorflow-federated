<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.from_keras_model" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.from_keras_model

```python
tff.learning.from_keras_model(
    keras_model,
    dummy_batch,
    loss,
    metrics=None,
    optimizer=None
)
```

Defined in
[`learning/model_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/model_utils.py).

<!-- Placeholder for "Used in" -->

Builds a
<a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a> for an
example mini batch.

#### Args:

*   <b>`keras_model`</b>: A `tf.keras.Model` object that is not compiled.
*   <b>`dummy_batch`</b>: A nested structure of values that are convertible to
    *batched* tensors with the same shapes and types as would be input to
    `keras_model`. The values of the tensors are not important and can be filled
    with any reasonable input value.
*   <b>`loss`</b>: A callable that takes two batched tensor parameters, `y_true`
    and `y_pred`, and returns the loss.
*   <b>`metrics`</b>: (Optional) a list of `tf.keras.metrics.Metric` objects.
*   <b>`optimizer`</b>: (Optional) a `tf.keras.optimizer.Optimizer`. If None,
    returned model cannot be used for training.

#### Returns:

A <a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a>
object.

#### Raises:

*   <b>`TypeError`</b>: If `keras_model` is not an instance of `tf.keras.Model`.
*   <b>`ValueError`</b>: If `keras_model` was compiled.
