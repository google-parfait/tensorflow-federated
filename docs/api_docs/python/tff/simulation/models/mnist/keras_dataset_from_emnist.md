<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.models.mnist.keras_dataset_from_emnist" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.models.mnist.keras_dataset_from_emnist

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/models/mnist.py">View
source</a>

Converts `dataset` for use with the output of `create_simple_keras_model`.

```python
tff.simulation.models.mnist.keras_dataset_from_emnist(dataset)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`dataset`</b>: An instance of `tf.data.Dataset` to read from.

#### Returns:

An instance of `tf.data.Dataset` after conversion.
