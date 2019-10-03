<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.HDF5ClientData" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="client_ids"/>
<meta itemprop="property" content="output_shapes"/>
<meta itemprop="property" content="output_types"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_tf_dataset_for_client"/>
<meta itemprop="property" content="create_tf_dataset_from_all_clients"/>
<meta itemprop="property" content="from_clients_and_fn"/>
<meta itemprop="property" content="preprocess"/>
</div>

# tff.simulation.HDF5ClientData

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/hdf5_client_data.py">View
source</a>

## Class `HDF5ClientData`

A
<a href="../../tff/simulation/ClientData.md"><code>tff.simulation.ClientData</code></a>
backed by an HDF5 file.

Inherits From: [`ClientData`](../../tff/simulation/ClientData.md)

<!-- Placeholder for "Used in" -->

This class expects that the HDF5 file has a top-level group `examples` which
contains further subgroups, one per user, named by the user ID.

The `tf.data.Dataset` returned by
<a href="../../tff/simulation/HDF5ClientData.md#create_tf_dataset_for_client"><code>HDF5ClientData.create_tf_dataset_for_client(client_id)</code></a>
yields tuples from zipping all datasets that were found at `/data/client_id`
group, in a similar fashion to `tf.data.Dataset.from_tensor_slices()`.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/hdf5_client_data.py">View
source</a>

```python
__init__(hdf5_filepath)
```

Constructs a
<a href="../../tff/simulation/ClientData.md"><code>tff.simulation.ClientData</code></a>
object.

#### Args:

*   <b>`hdf5_filepath`</b>: String path to the hdf5 file.

## Properties

<h3 id="client_ids"><code>client_ids</code></h3>

The list of string identifiers for clients in this dataset.

<h3 id="output_shapes"><code>output_shapes</code></h3>

Returns the shape of each component of an element of the client datasets.

Any `tf.data.Dataset` constructed by this class is expected to have matching
`output_shapes` properties when accessed via
`tf.compat.v1.data.get_output_shapes(dataset)`.

#### Returns:

A nested structure of `tf.TensorShape` objects corresponding to each component
of an element of the client datasets.

<h3 id="output_types"><code>output_types</code></h3>

Returns the type of each component of an element of the client datasets.

Any `tf.data.Dataset` constructed by this class is expected have matching
`output_types` properties when accessed via
`tf.compat.v1.data.get_output_types(dataset)`.

#### Returns:

A nested structure of `tf.DType` objects corresponding to each component of an
element of the client datasets.

## Methods

<h3 id="create_tf_dataset_for_client"><code>create_tf_dataset_for_client</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/hdf5_client_data.py">View
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

<h3 id="from_clients_and_fn"><code>from_clients_and_fn</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/client_data.py">View
source</a>

```python
from_clients_and_fn(
    cls,
    client_ids,
    create_tf_dataset_for_client_fn
)
```

Constructs a `ClientData` based on the given function.

#### Args:

*   <b>`client_ids`</b>: A non-empty list of client_ids which are valid inputs
    to the create_tf_dataset_for_client_fn.
*   <b>`create_tf_dataset_for_client_fn`</b>: A function that takes a client_id
    from the above list, and returns a `tf.data.Dataset`.

#### Returns:

A `ClientData`.

<h3 id="preprocess"><code>preprocess</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/client_data.py">View
source</a>

```python
preprocess(preprocess_fn)
```

Applies `preprocess_fn` to each client's data.
