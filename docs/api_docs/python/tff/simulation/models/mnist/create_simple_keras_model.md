<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.models.mnist.create_simple_keras_model" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.models.mnist.create_simple_keras_model

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/models/mnist.py">View
source</a>

Returns an instance of `tf.Keras.Model` with just one dense layer.

```python
tff.simulation.models.mnist.create_simple_keras_model(learning_rate=0.1)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`learning_rate`</b>: The learning rate to use with the SGD optimizer.

#### Returns:

An instance of `tf.Keras.Model`.
