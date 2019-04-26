<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.datasets.emnist.get_synthetic" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.datasets.emnist.get_synthetic

Quickly returns a small synthetic dataset, useful for unit tests, etc.

```python
tff.simulation.datasets.emnist.get_synthetic(num_clients=2)
```

Defined in
[`simulation/datasets/emnist/load_data.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/datasets/emnist/load_data.py).

<!-- Placeholder for "Used in" -->

Each client produced has exactly 10 examples, one of each digit. The images are
derived from a fixed set of hard-coded images, and transformed using
`tff.simulation.datasets.emnist.infinite_emnist` to produce the desired number
of clients.

#### Args:

*   <b>`num_clients`</b>: The number of syntehtic clients to generate.

#### Returns:

Tuple of (train, test) where the tuple elements are
<a href="../../../../tff/simulation/ClientData.md"><code>tff.simulation.ClientData</code></a>
objects matching the characteristics (other than size) of those provided by
<a href="../../../../tff/simulation/datasets/emnist/load_data.md"><code>tff.simulation.datasets.emnist.load_data</code></a>.
