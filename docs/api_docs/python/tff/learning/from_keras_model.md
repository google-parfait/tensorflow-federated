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

Builds a `tensorflow_federated.learning.Model`.

#### Args:

* <b>`keras_model`</b>: a `tf.keras.Model` object that is not compiled.
* <b>`loss`</b>: a callable that takes two batched tensor parameters, `y_true` and
    `y_pred`, and returns the loss.
* <b>`metrics`</b>: a list of `tf.keras.metrics.Metric` objects. The value of
    `Metric.result()` for each metric is included in the list of tensors
    returned in `aggregated_outputs()`.
* <b>`optimizer`</b>: a `tf.keras.optimizer.Optimizer`.


#### Returns:

A `tensorflow_federated.learning.TrainableModel` object iff optimizer is not
`None`, otherwise a `tensorflow_federated.learning.Model` object.


#### Raises:

* <b>`TypeError`</b>: if keras_model is not an instace of `tf.keras.Model`.
* <b>`ValueError`</b>: if keras_model was compiled.