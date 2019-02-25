<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.TransformingClientData" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="client_ids"/>
<meta itemprop="property" content="output_shapes"/>
<meta itemprop="property" content="output_types"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_tf_dataset_for_client"/>
<meta itemprop="property" content="create_tf_dataset_from_all_clients"/>
</div>

# tff.simulation.TransformingClientData

## Class `TransformingClientData`

Inherits From: [`ClientData`](../../tff/simulation/ClientData.md)

Defined in
[`simulation/transforming_client_data.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/transforming_client_data.py).

Expands client data by performing transformations.

Each client of the raw_client_data is "expanded" into some number of
pseudo-clients. Each client ID is a tuple containing the original client ID plus
an integer index. A function f(x, i) maps datapoints x with index i to new
datapoint. For example if x is an image, and i has values 0 or 1, f(x, 0) might
be the identity, while f(x, 1) could be the reflection of the image.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    raw_client_data,
    transform_fn,
    num_transformed_clients
)
```

Initializes the TransformingClientData.

#### Args:

*   <b>`raw_client_data`</b>: A ClientData to expand.
*   <b>`transform_fn`</b>: A function f(x, i) parameterized by i, mapping
    datapoint x to a new datapoint. x is a datapoint from the raw_client_data,
    while i is an integer index in the range 0...k (see
    'num_transformed_clients' for definition of k). Typically by convention the
    index 0 corresponds to the identity function if the identity is supported.
*   <b>`num_transformed_clients`</b>: The total number of transformed clients to
    produce. If it is an integer multiple k of the number of real clients, there
    will be exactly k pseudo-clients per real client, with indices 0...k-1. Any
    remainder g will be generated from the first g real clients and will be
    given index k.

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
