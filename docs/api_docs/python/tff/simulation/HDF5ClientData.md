<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.HDF5ClientData" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="client_ids"/>
<meta itemprop="property" content="output_shapes"/>
<meta itemprop="property" content="output_types"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_tf_dataset_for_client"/>
<meta itemprop="property" content="create_tf_dataset_from_all_clients"/>
</div>

# tff.simulation.HDF5ClientData

## Class `HDF5ClientData`

Inherits From: [`ClientData`](../../tff/simulation/ClientData.md)

Defined in
[`simulation/hdf5_client_data.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/hdf5_client_data.py).

A `tf.simulation.ClientData` backed by an HDF5 file.

This class expects that the HDF5 file has a top-level group `examples` which
contains further subgroups, one per user, named by the user ID.

The `tf.data.Dataset` returned by
`HDF5ClientData.create_tf_dataset_for_client(client_id)` yields tuples from
zipping all datasets that were found at `/data/client_id` group, in a similar
fashoin to `tf.data.Dataset.from_tensor_slices()`.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(hdf5_filepath)
```

Constructs a `tf.simulation.ClientData` object.

#### Args:

*   <b>`hdf5_filepath`</b>: String path to the hdf5 file.

## Properties

<h3 id="client_ids"><code>client_ids</code></h3>

<h3 id="output_shapes"><code>output_shapes</code></h3>

<h3 id="output_types"><code>output_types</code></h3>

## Methods

<h3 id="create_tf_dataset_for_client"><code>create_tf_dataset_for_client</code></h3>

```python
create_tf_dataset_for_client(client_id)
```

<h3 id="create_tf_dataset_from_all_clients"><code>create_tf_dataset_from_all_clients</code></h3>

```python
create_tf_dataset_from_all_clients()
```

Creates a new `tf.data.Dataset` containing _all_ client examples.

NOTE: the returned `tf.data.Dataset` is not serializable and runnable on other
devices, as it uses `tf.py_func` internally.

#### Returns:

A `tf.data.Dataset` object.
