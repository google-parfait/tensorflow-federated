# Copyright 2020, The TensorFlow Federated Authors.
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
"""A set of utilities for heavy hitter discovery."""

import collections
import statistics
import time

from absl import app
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_text as tf_text


@tf.function
def get_top_elements(dataset, max_user_contribution):
  """Gets the top max_user_contribution words from the input list.

  Note that the returned set of top words will not necessarily be sorted.

  Args:
    dataset: A `tf.data.Dataset` to extract top elements from.
    max_user_contribution: The maximum number of elements to keep.

  Returns:
    A tensor of a list of strings.
    If the total number of unique words is less than or equal to
    max_user_contribution, returns the set of unique words.
  """
  # Create a tuple of parallel elements and counts. This will be appended to and
  # updated as we iterate over the dataset.
  element_type = dataset.element_spec.dtype
  initial_histogram = (tf.constant([], dtype=element_type),
                       tf.constant([], dtype=tf.int64))

  def count_word(histogram, new_element):
    elements, counts = histogram
    mask = tf.equal(elements, new_element)
    # If the element doesn't match any we've already seen, expand the list of
    # elements we are tracking and add one for the count.
    if not tf.reduce_any(mask):
      elements = tf.concat(
          [elements, tf.expand_dims(new_element, axis=0)], axis=0)
      counts = tf.concat([counts, tf.constant([1], dtype=tf.int64)], axis=0)
    else:
      # Otherwise add one to the index that was `True`.
      counts += tf.cast(mask, tf.int64)
    return elements, counts

  words, counts = dataset.reduce(
      initial_state=initial_histogram, reduce_func=count_word)
  if tf.size(words) > max_user_contribution:
    # This logic is influenced by the focus on global heavy hitters and
    # thus implements clipping by chopping the tail of the distribution
    # of the words as present on a single client. Another option could
    # be to provide pick max_words_per_user random words out of the unique
    # words present locally.
    top_indices = tf.argsort(
        counts, axis=-1, direction='DESCENDING')[:max_user_contribution]
    top_words = tf.gather(words, top_indices)
    return top_words
  return words


@tf.function
def get_random_elements(list_of_elements, max_user_contribution):
  """Gets random max_user_contribution words from the input list.

  Args:
    list_of_elements: A tensor containing a list of elements.
    max_user_contribution: The maximum number of elements to keep.

  Returns:
    A tensor of a list of strings.
  """
  return tf.random.shuffle(list_of_elements)[:max_user_contribution]


@tf.function
def tokenize(ds, dataset_name):
  """Tokenizes a line into words with alphanum characters."""

  def extract_strings(example):
    if dataset_name == 'shakespeare':
      return tf.expand_dims(example['snippets'], 0)
    elif dataset_name == 'stackoverflow':
      return tf.expand_dims(example['tokens'], 0)
    else:
      raise app.UsageError('Dataset not supported: ', dataset_name)

  def tokenize_line(line):
    return tf.data.Dataset.from_tensor_slices(tokenizer.tokenize(line)[0])

  def mask_all_symbolic_words(word):
    return tf.math.logical_not(
        tf_text.wordshape(word, tf_text.WordShape.IS_PUNCT_OR_SYMBOL))

  tokenizer = tf_text.WhitespaceTokenizer()
  ds = ds.map(extract_strings)
  ds = ds.flat_map(tokenize_line)
  ds = ds.map(tf_text.case_fold_utf8)
  ds = ds.filter(mask_all_symbolic_words)
  return ds


def get_federated_tokenize_fn(dataset_name, dataset_element_type_structure):
  """Get a federated tokenizer function."""

  @tff.tf_computation(tff.SequenceType(dataset_element_type_structure))
  def tokenize_dataset(dataset):
    """The TF computation to tokenize a dataset."""
    dataset = tokenize(dataset, dataset_name)
    return dataset

  @tff.federated_computation(
      tff.FederatedType(
          tff.SequenceType(dataset_element_type_structure), tff.CLIENTS))
  def tokenize_datasets(datasets):
    """The TFF computation to compute tokenized datasets."""
    tokenized_datasets = tff.federated_map(tokenize_dataset, datasets)
    return tokenized_datasets

  return tokenize_datasets


def distance_l1(ground_truth, signal, correction=1.0):
  """Computes the L1 distance between two {'string': frequency} dicts.

  Args:
    ground_truth: The ground truth dict.
    signal: The obtained heavy hitters dict.
    correction: The correction to boost the signal. Often the ratio between the
      overall client population and the number of clients that participated in a
      round.

  Returns:
    The L1 distance between the signal and its ground truth.
  """
  joined = collections.defaultdict(float)
  for k, v in ground_truth.items():
    joined[k] += float(v)
  for k, v in signal.items():
    joined[k] -= float(correction) * float(v)
  total = 0
  for v in joined.values():
    total += abs(v)
  return total


def precision(ground_truth, signal, k):
  """Computes the precision for the top k words between frequency dicts.

  Args:
    ground_truth: The ground truth dict.
    signal: The obtained heavy hitters dict.
    k: The number of top items that are consider heavy hitters.

  Returns:
    Precision of the signal in detecting a top k item.
  """
  top_k_ground_truth = set(top_k(ground_truth, k).keys())
  top_k_signal = set(top_k(signal, k).keys())
  true_positives = len(top_k_signal.intersection(top_k_ground_truth))
  if top_k_signal:
    return float(true_positives) / len(top_k_signal)
  else:
    return 0.0


def recall(ground_truth, signal, k):
  """Computes the recall for the top k words between frequency dicts.

  Args:
    ground_truth: The ground truth dict.
    signal: The obtained heavy hitters dict.
    k: The number of top items that are consider heavy hitters.

  Returns:
    Recall of the signal in detecting a top k item.
  """
  top_k_ground_truth = set(top_k(ground_truth, k).keys())
  top_k_signal = set(top_k(signal, k).keys())
  true_positives = len(top_k_signal.intersection(top_k_ground_truth))
  if top_k_ground_truth:
    return float(true_positives) / len(top_k_ground_truth)
  else:
    return 0.0


def f1_score(ground_truth, signal, k):
  """Computes the f1 score for the top k words between frequency dicts.

  Args:
    ground_truth: The ground truth dict.
    signal: The obtained heavy hitters dict.
    k: The number of top items that are consider heavy hitters.

  Returns:
    F1 score of the signal in detecting a top k item.
  """
  prec = precision(ground_truth, signal, k)
  rec = recall(ground_truth, signal, k)
  return statistics.harmonic_mean([prec, rec])


def top_k(signal, k):
  """Computes the top_k cut of a {'string': frequency} dict."""
  counter = collections.Counter(signal)
  ranks = collections.defaultdict(list)
  for key, value in counter.most_common():
    ranks[value].append(key)
  results = {}
  counter = 0
  for freq, values in ranks.items():
    for v in values:
      results[v] = freq
      counter += 1
    if counter >= k:
      break
  return results


def compute_loss(results,
                 expected_results,
                 correction,
                 space=None,
                 space_cost_per_error=None,
                 factor_bandwidth_into_loss=False):
  """Computes the loss between results and expected_results."""
  distance = distance_l1(
      signal=results, ground_truth=expected_results, correction=correction)
  if factor_bandwidth_into_loss:
    distance = distance + (float(space) / space_cost_per_error)
  return distance


def enough_variation(new_results, old_results, min_variation):
  if not old_results or not new_results:
    return True
  new_keys = set(new_results.keys())
  old_keys = set(old_results.keys())
  intersection = new_keys - old_keys
  logging.info('Result variation: %s', intersection)
  return len(intersection) >= min_variation


def get_top_words(datasets, num_words_per_dataset=None):
  total_words = []
  for dataset in datasets:
    words = [word.numpy().decode('utf-8') for word in dataset]
    if num_words_per_dataset is not None:
      words = collections.Counter(words).most_common(num_words_per_dataset)
      words = [word_pair[0] for word_pair in words]
    total_words += list(set(words))
  return total_words


def compute_expected_results(datasets, limit):
  return top_k(get_top_words(datasets), limit)


def calculate_ground_truth(data, dataset_name):
  """Gets all the words in the entire dataset."""
  all_datasets = [
      tokenize(data.create_tf_dataset_for_client(client_id), dataset_name)
      for client_id in data.client_ids
  ]

  start = time.time()
  ground_truth_results = get_top_words(all_datasets)
  logging.info('Obtained ground truth in %.2f seconds', time.time() - start)
  return ground_truth_results


@tff.tf_computation(tff.SequenceType(tf.string))
def compute_lossless_result_per_user(dataset):
  k_words = get_top_elements(dataset, 10)
  return k_words


@tff.federated_computation(
    tff.FederatedType(tff.SequenceType(tf.string), tff.CLIENTS))
def compute_lossless_results_federated(datasets):
  words = tff.federated_map(compute_lossless_result_per_user, datasets)
  return words


def compute_lossless_results(datasets):
  all_words = tf.concat(compute_lossless_results_federated(datasets), axis=0)
  word, _, count = tf.unique_with_counts(all_words)
  return dict(zip(word.numpy(), count.numpy()))
