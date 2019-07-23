<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.ClientData" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="client_ids"/>
<meta itemprop="property" content="output_shapes"/>
<meta itemprop="property" content="output_types"/>
<meta itemprop="property" content="create_tf_dataset_for_client"/>
<meta itemprop="property" content="create_tf_dataset_from_all_clients"/>
</div>

# tff.simulation.ClientData

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/client_data.py">View
source</a>

## Class `ClientData`

Object to hold a dataset and a mapping of clients to examples.

<!-- Placeholder for "Used in" -->

## Properties

<h3 id="client_ids"><code>client_ids</code></h3>

The list of string identifiers for clients in this dataset.

<h3 id="output_shapes"><code>output_shapes</code></h3>

Returns the shape of each component of an element of the client datasets.

Any `tf.data.Dataset` constructed by this class is expected to have matching
`tf.data.Dataset.output_shapes` properties.

#### Returns:

A nested structure of `tf.TensorShape` objects corresponding to each component
of an element of the client datasets.

<h3 id="output_types"><code>output_types</code></h3>

Returns the type of each component of an element of the client datasets.

Any `tf.data.Dataset` constructed by this class is expected have matching
`tf.data.Dataset.output_types` properties.

#### Returns:

A nested structure of `tf.DType` objects corresponding to each component of an
element of the client datasets.

## Methods

<h3 id="create_tf_dataset_for_client"><code>create_tf_dataset_for_client</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/client_data.py">View
source</a>

```python
create_tf_dataset_for_client(client_id)
```

Creates a new `tf.data.Dataset` containing the client training examples.

#### Args:

*   <b>`client_id`</b>: The string client_id for the desired client.

#### Returns:

A `tf.data.Dataset` object.

<h3 id="create_tf_dataset_from_all_clients"><code>create_tf_dataset_from_all_clients</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/client_data.py">View
source</a>

```python
create_tf_dataset_from_all_clients(seed=None)
```

Creates a new `tf.data.Dataset` containing _all_ client examples.

NOTE: the returned `tf.data.Dataset` is not serializable and runnable on other
devices, as it uses `tf.py_func` internally.

Currently, the implementation produces a dataset that contains all examples from
a single client in order, and so generally additional shuffling should be
performed.

#### Args:

*   <b>`seed`</b>: Optional, a seed to determine the order in which clients are
    processed in the joined dataset.

#### Returns:

A `tf.data.Dataset` object.
