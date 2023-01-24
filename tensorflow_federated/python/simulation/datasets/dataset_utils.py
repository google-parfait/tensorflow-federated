# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tools for working with tf.data.Datasets."""

import tensorflow as tf


def build_dataset_mixture(a, b, a_probability, seed=None):
  """Build a new dataset that probabilistically returns examples.

  Args:
    a: The first `tf.data.Dataset`.
    b: The second `tf.data.Dataset`.
    a_probability: The `float` probability to select the next example from the
      `a` dataset.
    seed: An optional `int` seed for creating a `tf.random.Generator`. If set to
      `None`, the generator will be created via
      `tf.random.Generator.from_non_deterministic_state`. Note that setting a
      random seed alone is not enough to guarantee complete reproducibility of
      results. For example, random number generation is not guaranteed to be
      consistent across TensorFlow versions (see
      https://www.tensorflow.org/guide/versions#what_is_not_covered).

  Returns:
    A `tf.data.Dataset` that returns examples from dataset `a` with probability
    `a_probability`, and examples form dataset `b` with probability `(1 -
    a_probability)`. The dataset will yield the number of examples equal to the
    smaller of `a` or `b`.
  """
  if seed is not None:
    random_generator = tf.random.Generator.from_seed(seed)
  else:
    random_generator = tf.random.Generator.from_non_deterministic_state()

  def _random_pick_example(example_a, example_b):
    if (
        random_generator.uniform(shape=(), minval=0.0, maxval=1.0)
        < a_probability
    ):
      return example_a
    return example_b

  # Note: this consumes a batch from both `a` and `b` each iteration. If
  # generating either batch is expensive, this maybe resulting in a lot of
  # waste.
  return tf.data.Dataset.zip((a, b)).map(_random_pick_example)


def build_single_label_dataset(dataset, label_key, desired_label):
  """Build a new dataset that only yields examples with a particular label.

  This can be used for creating pathological non-iid (in label space) datasets.

  Args:
    dataset: The base `tf.data.Dataset` that yields examples that are structures
      of string key -> tensor value pairs.
    label_key: The `str` key that holds the label for the example.
    desired_label: The label value to restrict the resulting dataset to.

  Returns:
    A `tf.data.Dataset` that is composed of only examples that have a label
    matching `desired_label`.
  """

  @tf.function
  def _select_on_label(example):
    return example[label_key] == desired_label

  return dataset.filter(_select_on_label)


def build_synthethic_iid_datasets(
    client_data, client_dataset_size: int, shuffle_buffer_size: int = 10000
):
  """Constructs an iterable of IID clients from a `tff.simulation.datasets.ClientData`.

  The returned iterator yields a stream of `tf.data.Datsets` that approximates
  the true statistical IID setting with the entirety of `client_data`
  representing the global distribution. That is, we do not simply randomly
  distribute the data across some fixed number of clients, instead each dataset
  returned by the iterator samples independently from the entirety of
  `client_data` (so any example in `client_data` may be produced by any client).

  Args:
    client_data: A `tff.simulation.datasets.ClientData`.
    client_dataset_size: The size of the `tf.data.Dataset` to yield from the
      returned dataset.
    shuffle_buffer_size: Shuffling buffer size for the union of all data from
      input `client_data`.

  Returns:
    A `tf.data.Dataset` instance that yields iid client datasets sampled from
    the global distribution.
  """
  global_dataset = client_data.create_tf_dataset_from_all_clients()
  global_dataset = global_dataset.shuffle(
      buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True
  )
  global_dataset = global_dataset.repeat(None)  # Repeat forever
  return global_dataset.window(client_dataset_size)
