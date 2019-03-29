<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.datasets.emnist.infinite_emnist" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.datasets.emnist.infinite_emnist

```python
tff.simulation.datasets.emnist.infinite_emnist(
    emnist_client_data,
    num_pseudo_clients
)
```

Defined in
[`simulation/datasets/emnist/load_data.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/datasets/emnist/load_data.py).

Converts a Federated EMNIST dataset into an Infinite Federated EMNIST set.

Infinite Federated EMNIST expands each writer from the EMNIST dataset into some
number of pseudo-clients each of whose characters are the same but apply a fixed
random affine transformation to the original user's characters. The distribution
over affine transformation is approximately equivalent to the one described at
https://www.cs.toronto.edu/~tijmen/affNIST/. It applies the following
transformations in this order:

1.  A random rotation chosen uniformly between -20 and 20 degrees.
2.  A random shearing adding between -0.2 to 0.2 of the x coordinate to the y
    coordinate (after centering).
3.  A random scaling between 0.8 and 1.25 (sampled log uniformly).
4.  A random translation between -5 and 5 pixels in both the x and y axes.

#### Args:

*   <b>`emnist_client_data`</b>: The
    <a href="../../../../tff/simulation/ClientData.md"><code>tff.simulation.ClientData</code></a>
    to convert.
*   <b>`num_pseudo_clients`</b>: How many pseudo-clients to generate for each
    real client. Each pseudo-client is formed by applying a given random affine
    transformation to the characters written by a given real user. The first
    pseudo-client for a given user applies the identity transformation, so the
    original users are always included.

#### Returns:

An expanded
<a href="../../../../tff/simulation/ClientData.md"><code>tff.simulation.ClientData</code></a>.
