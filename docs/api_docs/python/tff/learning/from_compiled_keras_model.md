<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.from_compiled_keras_model" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.from_compiled_keras_model

``` python
tff.learning.from_compiled_keras_model(keras_model)
```

Defined in
[`learning/model_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/model_utils.py).

Builds a `tensorflow_federated.learning.TrainableModel`.

#### Args:

* <b>`keras_model`</b>: a `tf.keras.Model` object that was compiled.


#### Returns:

A `tensorflow_federated.learning.TrainableModel`.


#### Raises:

* <b>`TypeError`</b>: if keras_model is not an instace of `tf.keras.Model`.
* <b>`ValueError`</b>: if keras_model was not compiled.