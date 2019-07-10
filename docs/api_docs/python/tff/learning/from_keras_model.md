<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.from_keras_model" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.from_keras_model

Builds a
<a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a> for an
example mini batch.

```python
tff.learning.from_keras_model(
    keras_model,
    dummy_batch,
    loss,
    loss_weights=None,
    metrics=None,
    optimizer=None
)
```

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/keras_utils.py>View
source</a>

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`keras_model`</b>: A `tf.keras.Model` object that is not compiled.
*   <b>`dummy_batch`</b>: A nested structure of values that are convertible to
    *batched* tensors with the same shapes and types as would be input to
    `keras_model`. The values of the tensors are not important and can be filled
    with any reasonable input value.
*   <b>`loss`</b>: A callable that takes two batched tensor parameters, `y_true`
    and `y_pred`, and returns the loss. If the model has multiple outputs, you
    can use a different loss on each output by passing a dictionary or a list of
    losses. The loss value that will be minimized by the model will then be the
    sum of all individual losses, each weighted by `loss_weights`.
*   <b>`loss_weights`</b>: (Optional) a list or dictionary specifying scalar
    coefficients (Python floats) to weight the loss contributions of different
    model outputs. The loss value that will be minimized by the model will then
    be the *weighted sum* of all individual losses, weighted by the
    `loss_weights` coefficients. If a list, it is expected to have a 1:1 mapping
    to the model's outputs. If a tensor, it is expected to map output names
    (strings) to scalar coefficients.
*   <b>`metrics`</b>: (Optional) a list of `tf.keras.metrics.Metric` objects.
*   <b>`optimizer`</b>: (Optional) a `tf.keras.optimizer.Optimizer`. If None,
    returned model cannot be used for training.

#### Returns:

A <a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a>
object.

#### Raises:

*   <b>`TypeError`</b>: If `keras_model` is not an instance of `tf.keras.Model`.
*   <b>`ValueError`</b>: If `keras_model` was compiled.
*   <b>`KeyError`</b>: If `loss` is a `dict` and does not have the same keys as
    `keras_model.outputs`.
