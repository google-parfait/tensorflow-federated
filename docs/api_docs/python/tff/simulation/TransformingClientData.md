<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.TransformingClientData" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="client_ids"/>
<meta itemprop="property" content="output_shapes"/>
<meta itemprop="property" content="output_types"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create_tf_dataset_for_client"/>
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
    transform,
    expansion_factor
)
```

Initializes the TransformingClientData.

#### Args:

*   <b>`raw_client_data`</b>: A ClientData to expand.
*   <b>`transform`</b>: A function f(x, i) mapping datapoint x to a new
    datapoint parameterized by i.
*   <b>`expansion_factor`</b>: The (expected) number of transformed clients per
    raw client. If not an integer, each client is mapped to at least
    int(expansion_factor) new clients, and some fraction of clients are mapped
    to one more.

## Properties

<h3 id="client_ids"><code>client_ids</code></h3>

<h3 id="output_shapes"><code>output_shapes</code></h3>

<h3 id="output_types"><code>output_types</code></h3>

## Methods

<h3 id="create_tf_dataset_for_client"><code>create_tf_dataset_for_client</code></h3>

```python
create_tf_dataset_for_client(client_id)
```
