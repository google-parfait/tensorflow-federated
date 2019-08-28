<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.datasets.build_single_label_dataset" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.datasets.build_single_label_dataset

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/datasets/dataset_utils.py">View
source</a>

Build a new dataset that only yields examples with a particular label.

```python
tff.simulation.datasets.build_single_label_dataset(
    dataset,
    label_key,
    desired_label
)
```

<!-- Placeholder for "Used in" -->

This can be used for creating pathological non-iid (in label space) datasets.

#### Args:

*   <b>`dataset`</b>: the base `tf.data.Dataset` that yields examples that are
    structures of string key -> tensor value pairs.
*   <b>`label_key`</b>: the `str` key that holds the label for the example.
*   <b>`desired_label`</b>: the label value to restrict the resulting dataset
    to.

#### Returns:

A `tf.data.Dataset` that is composed of only examples that have a label matching
`desired_label`.
