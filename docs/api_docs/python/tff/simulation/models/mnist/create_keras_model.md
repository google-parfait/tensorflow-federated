<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.models.mnist.create_keras_model" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.models.mnist.create_keras_model

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/models/mnist.py">View
source</a>

Returns an instance of `tf.keras.Model` for use with the MNIST example.

```python
tff.simulation.models.mnist.create_keras_model(compile_model=False)
```

<!-- Placeholder for "Used in" -->

This code is based on the following target, which unfortunately cannot be
imported as it is a Python binary, not a library:

https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py

#### Args:

*   <b>`compile_model`</b>: If True, compile the model with a basic optimizer
    and loss.

#### Returns:

A `tf.keras.Model`.
