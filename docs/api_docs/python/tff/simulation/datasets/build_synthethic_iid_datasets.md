<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.datasets.build_synthethic_iid_datasets" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.datasets.build_synthethic_iid_datasets

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/datasets/dataset_utils.py">View
source</a>

Constructs an iterable of IID clients from a `tf.data.Dataset`.

```python
tff.simulation.datasets.build_synthethic_iid_datasets(
    client_data,
    client_dataset_size
)
```

<!-- Placeholder for "Used in" -->

The returned iterator yields a stream of `tf.data.Datsets` that approximates the
true statistical IID setting with the entirety of `client_data` representing the
global distribution. That is, we do not simply randomly distribute the data
across some fixed number of clients, instead each dataset returned by the
iterator samples independently from the entirety of `client_data` (so any
example in `client_data` may be produced by any client).

#### Args:

*   <b>`client_data`</b>: a
    <a href="../../../tff/simulation/ClientData.md"><code>tff.simulation.ClientData</code></a>.
*   <b>`client_dataset_size`</b>: the size of the `tf.data.Dataset` to yield
    from the returned dataset.

#### Returns:

A `tf.data.Dataset` instance that yields iid client datasets sampled from

the global distribution.
