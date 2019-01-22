<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.from_keras_model" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.from_keras_model

``` python
tff.learning.from_keras_model(
    keras_model,
    loss,
    metrics=None,
    optimizer=None
)
```



Defined in [`learning/model_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/model_utils.py).

Builds a <a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a>.

#### Args:

* <b>`keras_model`</b>: a `tf.keras.Model` object that is not compiled.
* <b>`loss`</b>: a callable that takes two batched tensor parameters, `y_true` and
    `y_pred`, and returns the loss.
* <b>`metrics`</b>: a list of `tf.keras.metrics.Metric` objects. The value of
    `Metric.result()` for each metric is included in the list of tensors
    returned in `aggregated_outputs()`.
* <b>`optimizer`</b>: a `tf.keras.optimizer.Optimizer`.


#### Returns:

A <a href="../../tff/learning/Model.md"><code>tff.learning.Model</code></a> object.


#### Raises:

* <b>`TypeError`</b>: if `keras_model` is not an instance of `tf.keras.Model`.
* <b>`ValueError`</b>: if `keras_model` was compiled.