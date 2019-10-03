<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.datasets.build_dataset_mixture" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.datasets.build_dataset_mixture

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/datasets/dataset_utils.py">View
source</a>

Build a new dataset that probabilistically returns examples.

```python
tff.simulation.datasets.build_dataset_mixture(
    a,
    b,
    a_probability
)
```

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`a`</b>: the first `tf.data.Dataset`.
*   <b>`b`</b>: the second `tf.data.Dataset`.
*   <b>`a_probability`</b>: the `float` probability to select the next example
    from the `a` dataset.

#### Returns:

A `tf.data.Dataset` that returns examples from dataset `a` with probability
`a_probability`, and examples form dataset `b` with probability `(1 -
a_probability)`. The dataset will yield the number of examples equal to the
smaller of `a` or `b`.
