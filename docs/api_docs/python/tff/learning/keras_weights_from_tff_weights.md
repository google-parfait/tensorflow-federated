<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.learning.keras_weights_from_tff_weights" />
<meta itemprop="path" content="Stable" />
</div>

# tff.learning.keras_weights_from_tff_weights

```python
tff.learning.keras_weights_from_tff_weights(tff_weights)
```

Defined in
[`learning/model_utils.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/learning/model_utils.py).

Converts TFF's nested weights structure to flat weights.

This function may be used, for example, to retrieve the model parameters trained
by the federated averaging process for use in an existing `keras` model, e.g.:

```
fed_avg = tff.learning.build_federated_averaging_process(...)
state = fed_avg.initialize()
state = fed_avg.next(state, ...)
...
keras_model.set_weights(
    tff.learning.keras_weights_from_tff_weights(state.model))
```

#### Args:

*   <b>`tff_weights`</b>: A TFF value representing the weights of a model.

#### Returns:

A list of tensors suitable for passing to `tf.keras.Model.set_weights`.
