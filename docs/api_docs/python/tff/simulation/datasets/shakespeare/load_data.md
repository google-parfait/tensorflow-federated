<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.datasets.shakespeare.load_data" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.datasets.shakespeare.load_data

Loads the federated Shakespeare dataset.

```python
tff.simulation.datasets.shakespeare.load_data(cache_dir=None)
```

Defined in
[`simulation/datasets/shakespeare/load_data.py`](http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/datasets/shakespeare/load_data.py).

<!-- Placeholder for "Used in" -->

Downloads and caches the dataset locally. If previously downloaded, tries to
load the dataset from cache.

This dataset is derived from the Leaf repository
(https://github.com/TalwalkarLab/leaf) pre-processing on the works of
Shakespeare, which is published in "LEAF: A Benchmark for Federated Settings"
https://arxiv.org/abs/1812.01097.

The data set consists of 715 users (characters of Shakespeare plays), where each
example corresponds to a contiguous set of lines spoken by the character in a
given play.

#### Data set sizes:

-   train: 16,068 examples
-   test: 2,356 examples

Rather than holding out specific users, each user's examples are split across
_train_ and _test_ so that all users have at least one example in _train_ and
one example in _test_. Characters that had less than 2 examples are excluded
from the data set.

The `tf.data.Datasets` returned by
<a href="../../../../tff/simulation/ClientData.md#create_tf_dataset_for_client"><code>tff.simulation.ClientData.create_tf_dataset_for_client</code></a>
will yield `collections.OrderedDict` objects at each iteration, with the
following keys and values:

-   `'snippets'`: a `tf.Tensor` with `dtype=tf.string`, the snippet of
    contiguous text.

#### Args:

*   <b>`cache_dir`</b>: (Optional) directory to cache the downloaded file. If
    `None`, caches in Keras' default cache directory.

#### Returns:

Tuple of (train, test) where the tuple elements are
<a href="../../../../tff/simulation/ClientData.md"><code>tff.simulation.ClientData</code></a>
objects.
