<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.datasets.stackoverflow.load_word_counts" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.datasets.stackoverflow.load_word_counts

Loads the word counts for the Stackoverflow dataset.

```python
tff.simulation.datasets.stackoverflow.load_word_counts(cache_dir=None)
```

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/datasets/stackoverflow/load_data.py">View
source</a>

<!-- Placeholder for "Used in" -->

#### Args:

*   <b>`cache_dir`</b>: (Optional) directory to cache the downloaded file. If
    `None`, caches in Keras' default cache directory.

#### Returns:

A collections.OrderedDict where the keys are string tokens, and the values are
the counts of unique users who have at least one example in the training set
containing that token in the body text.
