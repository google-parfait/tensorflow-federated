<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.assign_weights_to_keras_model" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.assign_weights_to_keras_model

Assigns a nested structure of TFF weights to a Keras model.

```python
tff.learning.assign_weights_to_keras_model(
    keras_model,
    tff_weights
)
```

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/keras_utils.py>View
source</a>

<!-- Placeholder for "Used in" -->

This function may be used to retrieve the model parameters trained by the
federated averaging process for use in an existing `tf.keras.models.Model`,
e.g.:

```
keras_model = tf.keras.models.Model(inputs=..., outputs=...)

def model_fn():
  return tff.learning.from_keras_model(keras_model)

fed_avg = tff.learning.build_federated_averaging_process(model_fn, ...)
state = fed_avg.initialize()
state = fed_avg.next(state, ...)
...
tff.learning.assign_weights_to_keras_model(state.model, keras_model)
```

#### Args:

*   <b>`keras_model`</b>: A `tf.keras.models.Model` instance to assign weights
    to.
*   <b>`tff_weights`</b>: A TFF value representing the weights of a model.

#### Raises:

*   <b>`TypeError`</b>: if `tff_weights` is not a TFF value, or `keras_model` is
    not a `tf.keras.models.Model` instance.
