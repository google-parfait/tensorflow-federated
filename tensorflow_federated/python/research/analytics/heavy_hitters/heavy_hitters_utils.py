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

import bisect
import collections
import statistics
import time

from absl import app
from absl import logging
from scipy import stats

import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_text as tf_text


# These are keys in the shakespeare dataset, so need to be kept in sync.
SELECTED_CLIENTS = [
    'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_FRANCIS',
    'PERICLES__PRINCE_OF_TYRE_PRIEST',
    'THE_TAMING_OF_THE_SHREW_CERES',
    'PERICLES__PRINCE_OF_TYRE_MARSHAL',
    'ALL_S_WELL_THAT_ENDS_WELL_LOVE',
    'THE_TRAGEDY_OF_KING_LEAR_MRS',
    'THE_TAMING_OF_THE_SHREW_IRIS',
    'THE_TRAGEDY_OF_KING_LEAR_MARIANA',
    'PERICLES__PRINCE_OF_TYRE_ROSS',
    'ALL_S_WELL_THAT_ENDS_WELL_SECOND_CITIZEN',
]


def shakespeare_deterministic_sampler(data):
  """Returns a deterministic sample.

  Args:
    data: a tff.simulation.ClientData object.

  Returns:
    list of tf.data.Datasets.
  """
  return [
      tokenize(data.create_tf_dataset_for_client(client_id), 'shakespeare')
      for client_id in SELECTED_CLIENTS
  ]


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
  """Computes the top k cut of a {'string': frequency} dict.

  Args:
    signal: A dictionary of heavy hitters with counts.
    k: The number of top items to return.

  Returns:
    A dictionary of size k, containing the heavy hitters with the highest k
    counts. Note that the keys are sorted alphabetically for items with tied
    values, so the returned results are always consistent for the same input
    dictionary.
  """
  # The key might be None.
  if None in signal:
    del signal[None]

  if len(signal) <= k:
    return signal

  # Sort the dictionary decreasingly by counts, then increasingly by order in
  # the alphabet.
  sorted_signal = sorted(signal.items(), key=lambda x: (-x[1], x[0]))
  return dict(sorted_signal[:k])


def compute_loss(results,
                 expected_results,
                 correction,
                 communication_cost=None,
                 communication_cost_per_error=None,
                 factor_bandwidth_into_loss=False):
  """Computes the loss between results and expected_results."""
  distance = distance_l1(
      signal=results, ground_truth=expected_results, correction=correction)
  if factor_bandwidth_into_loss:
    distance = distance + (
        float(communication_cost) / communication_cost_per_error)
  return distance


def enough_variation(new_results, old_results, min_variation):
  if not old_results or not new_results:
    return True
  new_keys = set(new_results.keys())
  old_keys = set(old_results.keys())
  intersection = new_keys - old_keys
  logging.info('Result variation: %s', intersection)
  return len(intersection) >= min_variation


def get_all_words_counts(datasets):
  total_words = []
  for dataset in datasets:
    words = [word.numpy().decode('utf-8') for word in dataset]
    total_words += list(set(words))
  return dict(collections.Counter(total_words))


def compute_expected_results(datasets, limit):
  return top_k(get_all_words_counts(datasets), limit)


def calculate_ground_truth(data, dataset_name):
  """Gets all the words in the entire dataset."""
  start = time.time()
  all_datasets = [
      tokenize(data.create_tf_dataset_for_client(client_id), dataset_name)
      for client_id in data.client_ids
  ]
  ground_truth_results = get_all_words_counts(all_datasets)
  logging.info('Obtained ground truth in %.2f seconds', time.time() - start)
  return ground_truth_results


def compute_threshold_leakage(ground_truth, signal, t):
  """Computes the threshold leakage of the least frequent words.

  A word is leaked at threshold `t` if it appears less than `t` times in
  `ground_truth` but appears in `signal`.

  The false positive rate (FPR) at a threshold `t` is defined as the number of
  leaked words divided by the number of words appearing less than `t` times in
  `ground_truth`.

  The false discovery rate (FDR) at a threshold `t` is defined as the number of
  leaked words divided by the size of `signal`.

  Args:
    ground_truth: The ground truth dict.
    signal: The obtained heavy hitters dict.
    t: Compute FPR, FDR and the harmonic mean of these two with a leak threshold
      from 1 to t.

  Returns:
    Three dictionaries: false_positive_rate, false_discovery_rate,
    harmonic_mean_fpr_fdr leakage at threshold from 1 to t.
  """

  false_positive_rate = {}
  false_discovery_rate = {}
  harmonic_mean_fpr_fdr = {}

  # Order the ground truth dictionary increasingly by counts for binary search.
  ground_truth = collections.OrderedDict(
      sorted(ground_truth.items(), key=lambda x: x[1]))
  ground_truth_words = list(ground_truth.keys())
  ground_truth_counts = list(ground_truth.values())

  leaked_words_candidates = set(signal.keys())
  bisect_upper_bound = len(ground_truth_counts)
  signal_size = len(signal)

  # Iterate the threshold from t to 1. Note that leaked words of threshold t-1
  # is a subset of leaked words of threshold k.
  for threshold in range(t, 0, -1):
    below_threshold_index = bisect.bisect_left(
        ground_truth_counts, threshold, lo=0, hi=bisect_upper_bound)
    words_below_threshold = set(ground_truth_words[:below_threshold_index])
    leaked_words = words_below_threshold.intersection(leaked_words_candidates)
    leaked_words_count = len(leaked_words)

    # If leaked_words_count > 0, then it must be below_threhold_index > 0 and
    # signal_size > 0, so there won't be a "divide by 0" error.
    if leaked_words_count > 0:
      false_positive_rate[
          threshold] = leaked_words_count / below_threshold_index
      false_discovery_rate[threshold] = leaked_words_count / signal_size
      harmonic_mean_fpr_fdr[threshold] = stats.hmean(
          [false_positive_rate[threshold], false_discovery_rate[threshold]])
    else:
      false_positive_rate[threshold] = 0.0
      false_discovery_rate[threshold] = 0.0
      harmonic_mean_fpr_fdr[threshold] = 0.0

    # The leaked_words in the next round must be a subset of this round.
    leaked_words_candidates = leaked_words
    bisect_upper_bound = below_threshold_index

  return false_positive_rate, false_discovery_rate, harmonic_mean_fpr_fdr


@tff.tf_computation(tff.SequenceType(tf.string))
def compute_lossless_result_per_user(dataset):
  # Do not have limit on each client's contribution in this case.
  k_words = get_top_elements(dataset, tf.constant(tf.int32.max))
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
