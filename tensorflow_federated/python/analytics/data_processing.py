# Copyright 2021, The TensorFlow Federated Authors.
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
"""A set of utility functions for data processing."""

import math

import tensorflow as tf


@tf.function
def get_all_elements(dataset):
  """Gets all the elements from the input dataset.

  The input `dataset` must yield batched 1-d tensors. This function reads each
  coordinate of the tensor as an individual element.

  Args:
    dataset: A `tf.data.Dataset`.

  Returns:
    A rank-1 Tensor containing the elements of the input dataset.
  """

  if dataset.element_spec.shape.rank != 1:
    raise ValueError('The shape of elements in `dataset` must be of rank 1, '
                     f' found rank = {dataset.element_spec.shape.rank}'
                     ' instead.')

  element_type = dataset.element_spec.dtype
  initial_list = tf.constant([], dtype=element_type)

  def add_element(element_list, element_batch):
    element_list = tf.concat([element_list, element_batch], axis=0)
    return element_list

  all_element_list = dataset.reduce(
      initial_state=initial_list, reduce_func=add_element)

  return all_element_list


@tf.function
def get_capped_elements(dataset,
                        max_user_contribution: int,
                        batch_size: int = 1):
  """Gets the first max_user_contribution words from the input dataset.

  The input `dataset` must yield batched 1-d tensors. This function reads each
  coordinate of the tensor as an individual element and caps the total number of
  elements to return. Note either none of the elements in one batch is added to
  the returned result, or all the elements are added. This means the length of
  the returned list of elements could be less than `max_user_contribution` when
  `dataset` is capped.

  Args:
    dataset: A `tf.data.Dataset`.
    max_user_contribution: The maximum number of elements to return.
    batch_size: The number of elements in each batch of `dataset`.

  Returns:
    A rank-1 Tensor containing the elements of the input dataset after being
    capped. If the total number of words is less than or equal to
    `max_user_contribution`, returns all the words in `dataset`.
  """

  if dataset.element_spec.shape.rank != 1:
    raise ValueError('The shape of elements in `dataset` must be of rank 1, '
                     f' found rank = {dataset.element_spec.shape.rank}'
                     ' instead.')

  if max_user_contribution < 1:
    raise ValueError('`max_user_contribution` must be at least 1.')

  if batch_size < 1:
    raise ValueError('`batch_size` must be at least 1.')

  capped_size = math.floor(max_user_contribution / batch_size)
  capped_dataset = dataset.take(capped_size)
  return get_all_elements(capped_dataset)


@tf.function
def get_unique_elements(dataset):
  """Gets the unique elements from the input `dataset`.

  The input `dataset` must yield batched 1-d tensors. This function reads each
  coordinate of the tensor as an individual element and return unique elements.

  Args:
    dataset: A `tf.data.Dataset`.

  Returns:
    A rank-1 Tensor containing the unique elements of the input dataset.
  """

  if dataset.element_spec.shape.rank != 1:
    raise ValueError('The shape of elements in `dataset` must be of rank 1, '
                     f' found rank = {dataset.element_spec.shape.rank}'
                     ' instead.')

  element_type = dataset.element_spec.dtype
  initial_list = tf.constant([], dtype=element_type)

  def add_unique_element(element_list, element_batch):
    element_list = tf.concat([element_list, element_batch], axis=0)
    element_list, _ = tf.unique(element_list)
    return element_list

  unique_element_list = dataset.reduce(
      initial_state=initial_list, reduce_func=add_unique_element)

  return unique_element_list


# TODO(b/192336690): Improve the efficiency of `get_top_elememnts`.
# The current implementation iterates `dataset` twice, which is not optimal.
@tf.function
def get_top_elements(dataset, max_user_contribution):
  """Gets the top `max_user_contribution` unique words from the input `dataset`.

  The input `dataset` must yield batched 1-d tensors. This function reads each
  coordinate of the tensor as an individual element and caps the total number of
  elements to return. Note that the returned set of top words will not
  necessarily be sorted.

  Args:
    dataset: A `tf.data.Dataset` to extract top elements from.
    max_user_contribution: The maximum number of elements to keep.

  Returns:
    A rank-1 Tensor containing the top `max_user_contribution` unique elements
    of the input `dataset`. If the total number of unique words is less than or
    equal to `max_user_contribution`, returns the list of all unique elements.
  """
  if dataset.element_spec.shape.rank != 1:
    raise ValueError('The shape of elements in `dataset` must be of rank 1, '
                     f' found rank = {dataset.element_spec.shape.rank}'
                     ' instead.')

  if max_user_contribution < 1:
    raise ValueError('`max_user_contribution` must be at least 1.')

  all_elements = get_all_elements(dataset)
  words, indices = tf.unique(all_elements)

  def accumulate_counts(counts, item):
    item_one_hot = tf.one_hot(item, tf.shape(words)[0], dtype=tf.int64)
    return counts + item_one_hot

  counts = tf.foldl(
      accumulate_counts,
      indices,
      initializer=tf.zeros_like(words, dtype=tf.int64))

  if tf.size(words) > max_user_contribution:
    top_indices = tf.argsort(
        counts, axis=-1, direction='DESCENDING')[:max_user_contribution]
    top_words = tf.gather(words, top_indices)
    return top_words
  return words
