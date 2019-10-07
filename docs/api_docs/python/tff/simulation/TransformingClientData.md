<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.TransformingClientData" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="client_ids"/>
<meta itemprop="property" content="output_shapes"/>
<meta itemprop="property" content="output_types"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_tf_dataset_for_client"/>
<meta itemprop="property" content="create_tf_dataset_from_all_clients"/>
<meta itemprop="property" content="from_clients_and_fn"/>
<meta itemprop="property" content="preprocess"/>
<meta itemprop="property" content="train_test_client_split"/>
</div>

# tff.simulation.TransformingClientData

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/transforming_client_data.py">View
source</a>

## Class `TransformingClientData`

Transforms client data, potentially expanding by adding pseudo-clients.

Inherits From: [`ClientData`](../../tff/simulation/ClientData.md)

<!-- Placeholder for "Used in" -->

Each client of the raw_client_data is "expanded" into some number of
pseudo-clients. Each client ID is a string consisting of the original client ID
plus a concatenated integer index. For example, the raw client id "client_a"
might be expanded into pseudo-client ids "client_a_0", "client_a_1" and
"client_a_2". A function fn(x) maps datapoint x to a new datapoint, where the
constructor of fn is parameterized by the (raw) client_id and index i. For
example if x is an image, then make_transform_fn("client_a", 0)(x) might be the
identity, while make_transform_fn("client_a", 1)(x) could be a random rotation
of the image with the angle determined by a hash of "client_a" and "1".
Typically by convention the index 0 corresponds to the identity function if the
identity is supported.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/transforming_client_data.py">View
source</a>

```python
__init__(
    raw_client_data,
    make_transform_fn,
    num_transformed_clients
)
```

Initializes the TransformingClientData.

#### Args:

*   <b>`raw_client_data`</b>: A ClientData to expand.
*   <b>`make_transform_fn`</b>: A function that returns a callable that maps
    datapoint x to a new datapoint x'. make_transform_fn will be called as
    make_transform_fn(raw_client_id, i) where i is an integer index, and should
    return a function fn(x)->x. For example if x is an image, then
    make_transform_fn("client_a", 0)(x) might be the identity, while
    make_transform_fn("client_a", 1)(x) could be a random rotation of the image
    with the angle determined by a hash of "client_a" and "1". If
    transform_fn_cons returns `None`, no transformation is performed. Typically
    by convention the index 0 corresponds to the identity function if the
    identity is supported.
*   <b>`num_transformed_clients`</b>: The total number of transformed clients to
    produce. If it is an integer multiple k of the number of real clients, there
    will be exactly k pseudo-clients per real client, with indices 0...k-1. Any
    remainder g will be generated from the first g real clients and will be
    given index k.

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

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/transforming_client_data.py">View
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

<h3 id="train_test_client_split"><code>train_test_client_split</code></h3>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/client_data.py">View
source</a>

```python
train_test_client_split(
    cls,
    client_data,
    num_test_clients
)
```

Returns a pair of (train, test) `ClientData`.

This method partitions the clients of `client_data` into two `ClientData`
objects with disjoint sets of
<a href="../../tff/simulation/ClientData.md#client_ids"><code>ClientData.client_ids</code></a>.
All clients in the test `ClientData` are guaranteed to have non-empty datasets,
but the training `ClientData` may have clients with no data.

Note: This method may be expensive, and so it may be useful to avoid calling
multiple times and holding on to the results.

#### Args:

*   <b>`client_data`</b>: The base `ClientData` to split.
*   <b>`num_test_clients`</b>: How many clients to hold out for testing. This
    can be at most len(client_data.client_ids) - 1, since we don't want to
    produce empty `ClientData`.

#### Returns:

A pair (train_client_data, test_client_data), where test_client_data has
`num_test_clients` selected at random, subject to the constraint they each have
at least 1 batch in their dataset.

#### Raises:

*   <b>`ValueError`</b>: If `num_test_clients` cannot be satistifed by
    `client_data`, or too many clients have empty datasets.
