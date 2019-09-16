<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.simulation.datasets.stackoverflow.load_data" />
<meta itemprop="path" content="Stable" />
</div>

# tff.simulation.datasets.stackoverflow.load_data

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="http://github.com/tensorflow/federated/tree/master/tensorflow_federated/python/simulation/datasets/stackoverflow.py">View
source</a>

Loads the federated Stackoverflow dataset.

```python
tff.simulation.datasets.stackoverflow.load_data(cache_dir=None)
```

<!-- Placeholder for "Used in" -->

Downloads and caches the dataset locally. If previously downloaded, tries to
load the dataset from cache.

This dataset is derived from the Stack Overflow Data hosted by kaggle.com and
available to query through Kernels using the BigQuery API:
https://www.kaggle.com/stackoverflow/stackoverflow. The Stack Overflow Data is
licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License.
To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/3.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.

The data consists of the body text of all questions and answers. The bodies were
parsed into sentences, and any user with fewer than 100 sentences was expunged
from the data. Minimal preprocessing was performed as follows:

1.  Lowercase the text,
2.  Unescape HTML symbols,
3.  Remove non-ascii symbols,
4.  Separate punctuation as individual tokens (except apostrophes and hyphens),
5.  Removing extraneous whitespace,
6.  Replacing URLS with a special token.

In addition the following metadata is available:

1.  Creation date
2.  Question title
3.  Question tags
4.  Question score
5.  Type ('question' or 'answer')

The data is divided into three sets:

-   Train: Data before 2018-01-01 UTC except the held-out users. 342,477 unique
    users with 135,818,730 examples.
-   Held-out: All examples from users with user_id % 10 == 0 (all dates). 38,758
    unique users with 16,491,230 examples.
-   Test: All examples after 2018-01-01 UTC except from held-out users. 204,088
    unique users with 16,586,035 examples.

The `tf.data.Datasets` returned by
<a href="../../../../tff/simulation/ClientData.md#create_tf_dataset_for_client"><code>tff.simulation.ClientData.create_tf_dataset_for_client</code></a>
will yield `collections.OrderedDict` objects at each iteration, with the
following keys and values:

-   `'creation_date'`: a `tf.Tensor` with `dtype=tf.string` and shape [1]
    containing the date/time of the question or answer in UTC format.
-   `'title'`: a `tf.Tensor` with `dtype=tf.string` and shape [1] containing the
    title of the question.
-   `'score'`: a `tf.Tensor` with `dtype=tf.int64` and shape [1] containing the
    score of the question.
-   `'tags'`: a `tf.Tensor` with `dtype=tf.string` and shape [1] containing the
    tags of the question, separated by '|' characters.
-   `'tokens'`: a `tf.Tensor` with `dtype=tf.string` and shape [1] containing
    the tokens of the question/answer, separated by space (' ') characters.
-   `'type'`: a `tf.Tensor` with `dtype=tf.string` and shape [1] containing
    either the string 'question' or 'answer'.

#### Args:

*   <b>`cache_dir`</b>: (Optional) directory to cache the downloaded file. If
    `None`, caches in Keras' default cache directory.

#### Returns:

Tuple of (train, held_out, test) where the tuple elements are
<a href="../../../../tff/simulation/ClientData.md"><code>tff.simulation.ClientData</code></a>
objects.
