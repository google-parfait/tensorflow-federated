<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.ClientData" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="client_ids"/>
<meta itemprop="property" content="output_shapes"/>
<meta itemprop="property" content="output_types"/>
<meta itemprop="property" content="create_tf_dataset_for_client"/>
</div>

# tff.simulation.ClientData

## Class `ClientData`





Defined in [`simulation/client_data.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/client_data.py).

Object to hold a dataset and a mapping of clients to examples.

## Properties

<h3 id="client_ids"><code>client_ids</code></h3>

The list of identifiers for clients in this dataset.

A client identifier can be any type understood by the
<a href="../../tff/simulation/ClientData.md#create_tf_dataset_for_client"><code>tff.simulation.ClientData.create_tf_dataset_for_client</code></a> method, determined
by the implementation.

<h3 id="output_shapes"><code>output_shapes</code></h3>

Returns the shape of each component of an element of the client datasets.

Any `tf.data.Dataset` constructed by this class is expected to have matching
`output_shapes` properties.

#### Returns:

  A nested structure of `tf.TensorShape` objects corresponding to each
component of an element of the client datasets.

<h3 id="output_types"><code>output_types</code></h3>

Returns the type of each component of an element of the client datasets.

Any `tf.data.Dataset` constructed by this class is expected have matching
`output_types` properties.

#### Returns:

  A nested structure of `tf.DType` objects corresponding to each component
of an element of the client datasets.



## Methods

<h3 id="create_tf_dataset_for_client"><code>create_tf_dataset_for_client</code></h3>

``` python
create_tf_dataset_for_client(client_id)
```

Creates a new `tf.data.Dataset` containing the client training examples.

#### Args:

* <b>`client_id`</b>: The identifier for the desired client.


#### Returns:

A `tf.data.Dataset` object.



