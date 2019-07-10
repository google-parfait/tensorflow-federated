<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.datasets.emnist.load_data" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.datasets.emnist.load_data

Loads the Federated EMNIST dataset.

```python
tff.simulation.datasets.emnist.load_data(
    only_digits=True,
    cache_dir=None
)
```

<a target="_blank" href=http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/datasets/emnist/load_data.py>View
source</a>

<!-- Placeholder for "Used in" -->

Downloads and caches the dataset locally. If previously downloaded, tries to
load the dataset from cache.

This dataset is derived from the Leaf repository
(https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
dataset, grouping examples by writer. Details about Leaf were published in
"LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.

*Note*: This dataset does not include some additional preprocessing that MNIST
includes, such as size-normalization and centering. In the Federated EMNIST
data, the value of 1.0 corresponds to the background, and 0.0 corresponds to the
color of the digits themselves; this is the *inverse* of some MNIST
representations, e.g. in
[tensorflow_datasets](https://github.com/tensorflow/datasets/blob/master/docs/datasets.md#mnist),
where 0 corresponds to the background color, and 255 represents the color of the
digit.

#### Data set sizes:

*only_digits=True*: 3,383 users, 10 label classes

-   train: 341,873 examples
-   test: 40,832 examples

*only_digits=False*: 3,400 users, 62 label classes

-   train: 671,585 examples
-   test: 77,483 examples

Rather than holding out specific users, each user's examples are split across
_train_ and _test_ so that all users have at least one example in _train_ and
one example in _test_. Writers that had less than 2 examples are excluded from
the data set.

The `tf.data.Datasets` returned by
<a href="../../../../tff/simulation/ClientData.md#create_tf_dataset_for_client"><code>tff.simulation.ClientData.create_tf_dataset_for_client</code></a>
will yield `collections.OrderedDict` objects at each iteration, with the
following keys and values:

-   `'pixels'`: a `tf.Tensor` with `dtype=tf.float32` and shape [28, 28],
    containing the pixels of the handwritten digit, with values in the range
    [0.0, 1.0].
-   `'label'`: a `tf.Tensor` with `dtype=tf.int32` and shape [1], the class
    label of the corresponding pixels.

#### Args:

*   <b>`only_digits`</b>: (Optional) whether to only include examples that are
    from the digits [0-9] classes. If `False`, includes lower and upper case
    characters, for a total of 62 class labels.
*   <b>`cache_dir`</b>: (Optional) directory to cache the downloaded file. If
    `None`, caches in Keras' default cache directory.

#### Returns:

Tuple of (train, test) where the tuple elements are
<a href="../../../../tff/simulation/ClientData.md"><code>tff.simulation.ClientData</code></a>
objects.
