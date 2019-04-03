<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.FilePerUserClientData" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="client_ids"/>
<meta itemprop="property" content="output_shapes"/>
<meta itemprop="property" content="output_types"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_from_dir"/>
<meta itemprop="property" content="create_tf_dataset_for_client"/>
<meta itemprop="property" content="create_tf_dataset_from_all_clients"/>
</div>

# tff.simulation.FilePerUserClientData

## Class `FilePerUserClientData`

Inherits From: [`ClientData`](../../tff/simulation/ClientData.md)

Defined in
[`simulation/file_per_user_client_data.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/file_per_user_client_data.py).

<!-- Placeholder for "Used in" -->

A `tf.simulation.ClientData` that maps a set of files to a dataset.

This mapping is restricted to one file per user.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    client_ids,
    create_tf_dataset_fn
)
```

Constructs a `tf.simulation.ClientData` object.

#### Args:

*   <b>`client_ids`</b>: A list of `client_id`s.
*   <b>`create_tf_dataset_fn`</b>: A callable that takes a `client_id` and
    returns a `tf.data.Dataset` object.

## Properties

<h3 id="client_ids"><code>client_ids</code></h3>

<h3 id="output_shapes"><code>output_shapes</code></h3>

<h3 id="output_types"><code>output_types</code></h3>

## Methods

<h3 id="create_from_dir"><code>create_from_dir</code></h3>

```python
@classmethod
create_from_dir(
    cls,
    path,
    create_tf_dataset_fn=tf.data.TFRecordDataset
)
```

Builds a
<a href="../../tff/simulation/FilePerUserClientData.md"><code>tff.simulation.FilePerUserClientData</code></a>.

Iterates over all files in `path`, using the filename as the client ID. Does not
recursively search `path`.

#### Args:

*   <b>`path`</b>: A directory path to search for per-client files.
*   <b>`create_tf_dataset_fn`</b>: A callable that creates a `tf.data.Datasaet`
    object for a given file in the directory specified in `path`.

#### Returns:

A
<a href="../../tff/simulation/FilePerUserClientData.md"><code>tff.simulation.FilePerUserClientData</code></a>
object.

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
