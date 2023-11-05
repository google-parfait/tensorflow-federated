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

from typing import Optional
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck


@tf.function
def get_all_elements(
    dataset: tf.data.Dataset, string_max_bytes: Optional[int] = None
):
  """Gets all the elements from the input dataset.

  The input `dataset` must yield batched rank-1 tensors. This function reads
  each coordinate of the tensor as an individual element.

  Args:
    dataset: A `tf.data.Dataset`.
    string_max_bytes: The maximum length (in bytes) of strings in the dataset.
      Strings longer than `string_max_bytes` will be truncated. Defaults to
      `None`, which means there is no limit of the string length.

  Returns:
    A rank-1 Tensor containing the elements of the input dataset.

  Raises:
    ValueError:
      -- If the shape of elements in `dataset` is not rank 1.
      -- If `string_max_bytes` is not `None` and is less than 1.
    TypeError: If `dataset.element_spec.dtype` must be `tf.string` is not
      `tf.string`.
  """

  if dataset.element_spec.shape.rank != 1:
    raise ValueError(
        'The shape of elements in `dataset` must be of rank 1, '
        f' found rank = {dataset.element_spec.shape.rank}'
        ' instead.'
    )

  if dataset.element_spec.dtype != tf.string:
    raise TypeError(
        '`dataset.element_spec.dtype` must be `tf.string`, found'
        f' element type {dataset.element_spec.dtype}'
    )

  if string_max_bytes is not None and string_max_bytes < 1:
    raise ValueError(
        '`string_max_bytes` must be at least 1 when it is not None.'
    )

  initial_list = tf.constant([], dtype=tf.string)

  def add_element(element_list, element_batch):
    if string_max_bytes is not None:
      element_batch = tf.strings.substr(
          element_batch, 0, string_max_bytes, unit='BYTE'
      )
    element_list = tf.concat([element_list, element_batch], axis=0)
    return element_list

  all_element_list = dataset.reduce(
      initial_state=initial_list, reduce_func=add_element
  )

  return all_element_list


def _get_capped_dataset(
    dataset: tf.data.Dataset, max_user_contribution: int, batch_size: int = 1
):
  """Returns capped `tf.data.Dataset` with the input `dataset`.

  The input `dataset` must yield batched rank-1 tensors. This function caps the
  number of elements in the dataset. Note either none of the elements in one
  batch is added to the returned result, or all the elements are added. This
  means the total number of elements in the returned dataset could be less than
  `max_user_contribution`.

  Args:
    dataset: A `tf.data.Dataset`.
    max_user_contribution: The maximum number of elements to return.
    batch_size: The number of elements in each batch of `dataset`.

  Returns:
    A `tf.data.Dataset` that contains at most the first `max_user_contribution`
      elements in the input `dataset`.

  Raises:
    ValueError:
      -- If the shape of elements in `dataset` is not rank 1.
      -- If `max_user_contribution` is less than 1.
      -- If `batch_size` is less than 1.
    TypeError: If `dataset.element_spec.dtype` must be `tf.string` is not
      `tf.string`.
  """
  if dataset.element_spec.shape.rank != 1:
    raise ValueError(
        'The shape of elements in `dataset` must be of rank 1, '
        f' found rank = {dataset.element_spec.shape.rank}'
        ' instead.'
    )

  if max_user_contribution < 1:
    raise ValueError('`max_user_contribution` must be at least 1.')

  if batch_size < 1:
    raise ValueError('`batch_size` must be at least 1.')

  if dataset.element_spec.dtype != tf.string:
    raise TypeError(
        '`dataset.element_spec.dtype` must be `tf.string`, found'
        f' element type {dataset.element_spec.dtype}'
    )

  capped_size = max_user_contribution // batch_size
  capped_dataset = dataset.take(capped_size)
  return capped_dataset


def get_capped_elements(
    dataset: tf.data.Dataset,
    max_user_contribution: int,
    batch_size: int = 1,
    string_max_bytes: Optional[int] = None,
):
  """Gets the first `max_user_contribution` elements from the input dataset.

  The input `dataset` must yield batched rank-1 tensors. This function reads
  each coordinate of the tensor as an individual element and caps the total
  number of elements to return. Note either none of the elements in one batch is
  added to the returned result, or all the elements are added. This means the
  length of the returned list of elements could be less than
  `max_user_contribution` when `dataset` is capped.

  Args:
    dataset: A `tf.data.Dataset`.
    max_user_contribution: The maximum number of elements to return.
    batch_size: The number of elements in each batch of `dataset`.
    string_max_bytes: The maximum length (in bytes) of strings in the dataset.
      Strings longer than `string_max_bytes` will be truncated. Defaults to
      `None`, which means there is no limit of the string length.

  Returns:
    A rank-1 Tensor containing the elements of the input dataset after being
    capped. If the total number of elements is less than or equal to
    `max_user_contribution`, returns all the elements in `dataset`.

  Raises:
    ValueError:
      -- If the shape of elements in `dataset` is not rank 1.
      -- If `max_user_contribution` is less than 1.
      -- If `batch_size` is less than 1.
      -- If `string_max_bytes` is not `None` and is less than 1.
    TypeError: If `dataset.element_spec.dtype` must be `tf.string` is not
      `tf.string`.
  """

  if string_max_bytes is not None and string_max_bytes < 1:
    raise ValueError(
        '`string_max_bytes` must be at least 1 when it is not None.'
    )
  capped_dataset = _get_capped_dataset(
      dataset=dataset,
      max_user_contribution=max_user_contribution,
      batch_size=batch_size,
  )
  return get_all_elements(capped_dataset, string_max_bytes)


def get_capped_elements_with_counts(
    dataset: tf.data.Dataset,
    max_user_contribution: int,
    batch_size: int = 1,
    string_max_bytes: Optional[int] = None,
):
  """Gets the capped elements with counts from the input dataset.

  The input `dataset` must yield batched rank-1 tensors. This function reads
  each coordinate of the tensor as an individual element and caps the total
  number of elements to return. Note either none of the elements in one batch is
  added to the returned result, or all the elements are added. This means the
  length of the returned list of elements could be less than
  `max_user_contribution` when `dataset` is capped.

  Args:
    dataset: A `tf.data.Dataset`.
    max_user_contribution: The maximum number of elements to return.
    batch_size: The number of elements in each batch of `dataset`.
    string_max_bytes: The maximum length (in bytes) of strings in the dataset.
      Strings longer than `string_max_bytes` will be truncated. Defaults to
      `None`, which means there is no limit of the string length.

  Returns:
    elements: A rank-1 Tensor containing the unique elements of the input
    dataset after being capped. If the total number of elements is less than or
    equal to `max_user_contribution`, returns all the elements in `dataset`.
    counts: A rank-1 Tensor containing the counts for each of the elements in
      `elements`.

  Raises:
    ValueError:
      -- If the shape of elements in `dataset` is not rank 1.
      -- If `max_user_contribution` is less than 1.
      -- If `batch_size` is less than 1.
      -- If `string_max_bytes` is not `None` and is less than 1.
    TypeError: If `dataset.element_spec.dtype` must be `tf.string` is not
      `tf.string`.
  """
  if string_max_bytes is not None and string_max_bytes < 1:
    raise ValueError(
        '`string_max_bytes` must be at least 1 when it is not None.'
    )
  capped_dataset = _get_capped_dataset(
      dataset=dataset,
      max_user_contribution=max_user_contribution,
      batch_size=batch_size,
  )
  return get_unique_elements_with_counts(capped_dataset, string_max_bytes)


@tf.function
def get_unique_elements(
    dataset: tf.data.Dataset, string_max_bytes: Optional[int] = None
):
  """Gets the unique elements from the input `dataset`.

  The input `dataset` must yield batched rank-1 tensors. This function reads
  each coordinate of the tensor as an individual element and return unique
  elements.

  Args:
    dataset: A `tf.data.Dataset`. Element type must be `tf.string`.
    string_max_bytes: The maximum length (in bytes) of strings in the dataset.
      Strings longer than `string_max_bytes` will be truncated. Defaults to
      `None`, which means there is no limit of the string length.

  Returns:
    A rank-1 Tensor containing the unique elements of the input dataset.

  Raises:
    ValueError:
      -- If the shape of elements in `dataset` is not rank 1.
      -- If `string_max_bytes` is not `None` and is less than 1.
    TypeError: If `dataset.element_spec.dtype` must be `tf.string` is not
      `tf.string`.
  """

  if dataset.element_spec.shape.rank != 1:
    raise ValueError(
        'The shape of elements in `dataset` must be of rank 1, '
        f' found rank = {dataset.element_spec.shape.rank}'
        ' instead.'
    )

  if string_max_bytes is not None and string_max_bytes < 1:
    raise ValueError(
        '`string_max_bytes` must be at least 1 when it is not None.'
    )

  if dataset.element_spec.dtype != tf.string:
    raise TypeError(
        '`dataset.element_spec.dtype` must be `tf.string`, found'
        f' element type {dataset.element_spec.dtype}'
    )

  initial_list = tf.constant([], dtype=tf.string)

  def add_unique_element(element_list, element_batch):
    if string_max_bytes is not None:
      element_batch = tf.strings.substr(
          element_batch, 0, string_max_bytes, unit='BYTE'
      )
    element_list = tf.concat([element_list, element_batch], axis=0)
    element_list, _ = tf.unique(element_list)
    return element_list

  unique_element_list = dataset.reduce(
      initial_state=initial_list, reduce_func=add_unique_element
  )

  return unique_element_list


# TODO: b/192336690 - Improve the efficiency of get_unique_elements_with_counts.
# The current implementation iterates `dataset` twice, which is not optimal.
def get_unique_elements_with_counts(
    dataset: tf.data.Dataset, string_max_bytes: Optional[int] = None
) -> tuple[tf.Tensor, tf.Tensor]:
  """Gets unique elements and their counts from the input `dataset`.

  This method returns a tuple of `elements` and `counts`, where `elements` are
  the unique elements in the dataset, and counts is the number of times each one
  appears.

  The input `dataset` must yield batched rank-1 tensors. This function reads
  each coordinate of the tensor as an individual element and caps the total
  number of elements to return.

  Args:
    dataset: A `tf.data.Dataset` to elements from. Element type must be
      `tf.string`.
    string_max_bytes: The maximum length (in bytes) of strings in the dataset.
      Strings longer than `string_max_bytes` will be truncated. Defaults to
      `None`, which means there is no limit of the string length.

  Returns:
    elements: A rank-1 Tensor containing all the unique elements of the input
    `dataset`.
    counts: A rank-1 Tensor containing the counts for each of the elements in
      `elements`.
  Raises:
    ValueError:
      -- If the shape of elements in `dataset` is not rank 1
      -- If `string_max_bytes` is not `None` and is less than 1.
    TypeError: If `dataset.element_spec.dtype` must be `tf.string` is not
      `tf.string`.
  """
  if dataset.element_spec.shape.rank != 1:
    raise ValueError(
        'The shape of elements in `dataset` must be of rank 1, '
        f' found rank = {dataset.element_spec.shape.rank}'
        ' instead.'
    )

  if string_max_bytes is not None and string_max_bytes < 1:
    raise ValueError(
        '`string_max_bytes` must be at least 1 when it is not None.'
    )

  if dataset.element_spec.dtype != tf.string:
    raise TypeError(
        '`dataset.element_spec.dtype` must be `tf.string`, found'
        f' element type {dataset.element_spec.dtype}'
    )

  all_elements = get_all_elements(dataset, string_max_bytes)

  elements, indices = tf.unique(all_elements)

  def accumulate_counts(counts, item):
    item_one_hot = tf.one_hot(item, tf.shape(elements)[0], dtype=tf.int64)
    return counts + item_one_hot

  counts = tf.foldl(
      accumulate_counts,
      indices,
      initializer=tf.zeros_like(elements, dtype=tf.int64),
  )

  return elements, counts


@tf.function
def get_top_elements_with_counts(
    dataset: tf.data.Dataset,
    max_user_contribution: int,
    string_max_bytes: Optional[int] = None,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Gets top unique elements from the input `dataset`.

  This method returns a tuple of `elements` and `counts`, where `elements` are
  the most common unique elements in the dataset, and counts is the number of
  times each one appears. The input `dataset` must yield batched rank-1 tensors.
  This function reads each coordinate of the tensor as an individual element and
  caps the total number of elements to return. Note that the returned set of top
  elements will not necessarily be sorted.

  Args:
    dataset: A `tf.data.Dataset` to extract top elements from. Element type must
      be `tf.string`.
    max_user_contribution: The maximum number of elements to keep.
    string_max_bytes: The maximum length (in bytes) of strings in the dataset.
      Strings longer than `string_max_bytes` will be truncated. Defaults to
      `None`, which means there is no limit of the string length.

  Returns:
    elements: A rank-1 Tensor containing the top `max_user_contribution` unique
      elements of the input `dataset`. If the total number of unique elements is
      less than or equal to `max_user_contribution`, returns the list of all
      unique elements.
    counts: A rank-1 Tensor containing the counts for each of the elements in
      `elements`.
  Raises:
    ValueError:
      -- If the shape of elements in `dataset` is not rank 1.
      -- If `max_user_contribution` is less than 1.
      -- If `string_max_bytes` is not `None` and is less than 1.
    TypeError: If `dataset.element_spec.dtype` must be `tf.string` is not
      `tf.string`.
  """
  if max_user_contribution < 1:
    raise ValueError('`max_user_contribution` must be at least 1.')

  elements, counts = get_unique_elements_with_counts(dataset, string_max_bytes)

  if tf.math.greater(tf.size(elements), max_user_contribution):
    counts, top_indices = tf.math.top_k(
        counts, max_user_contribution, sorted=False
    )
    top_elements = tf.gather(elements, top_indices)
    return top_elements, counts
  return elements, counts


def get_top_elements(
    dataset: tf.data.Dataset,
    max_user_contribution: int,
    string_max_bytes: Optional[int] = None,
):
  """Gets top unique elements from the input `dataset`.

  This method returns the set of `max_user_contribution` elements that appear
  most frequently in the dataset. Each word will only appear at most once in the
  output.

  This differs from `get_top_multi_elements` in that it returns a set rather
  than a multiset.

  The input `dataset` must yield batched rank-1 tensors. This function reads
  each coordinate of the tensor as an individual element and caps the total
  number of elements to return. Note that the returned set of top elements will
  not necessarily be sorted.

  Args:
    dataset: A `tf.data.Dataset` to extract top elements from. Element type must
      be `tf.string`.
    max_user_contribution: The maximum number of elements to keep.
    string_max_bytes: The maximum length (in bytes) of strings in the dataset.
      Strings longer than `string_max_bytes` will be truncated. Defaults to
      `None`, which means there is no limit of the string length.

  Returns:
    A rank-1 Tensor containing the top `max_user_contribution` unique elements
    of the input `dataset`. If the total number of unique elements is less than
    or equal to `max_user_contribution`, returns the list of all unique
    elements.

  Raises:
    ValueError:
      -- If the shape of elements in `dataset` is not rank 1.
      -- If `max_user_contribution` is less than 1.
      -- If `string_max_bytes` is not `None` and is less than 1.
    TypeError: If `dataset.element_spec.dtype` must be `tf.string` is not
      `tf.string`.
  """
  top_elements, _ = get_top_elements_with_counts(
      dataset=dataset,
      max_user_contribution=max_user_contribution,
      string_max_bytes=string_max_bytes,
  )
  return top_elements


def get_top_multi_elements(
    dataset: tf.data.Dataset,
    max_user_contribution: int,
    string_max_bytes: Optional[int] = None,
):
  """Gets the top unique word multiset from the input `dataset`.

  This method returns the `max_user_contribution` most common unique elements
  from the dataset, but returns a multiset. That is, a word will appear in the
  output as many times as it did in the dataset, but each unique word only
  counts one toward the `max_user_contribution` limit.

  This differs from `get_top_elements` in that it returns a multiset rather than
  a set.

  The input `dataset` must yield batched rank-1 tensors. This function reads
  each coordinate of the tensor as an individual element and caps the total
  number of elements to return. Note that the returned set of top elements will
  not necessarily be sorted.

  Args:
    dataset: A `tf.data.Dataset` to extract top elements from. Element type must
      be `tf.string`.
    max_user_contribution: The maximum number of elements to keep.
    string_max_bytes: The maximum length (in bytes) of strings in the dataset.
      Strings longer than `string_max_bytes` will be truncated. Defaults to
      `None`, which means there is no limit of the string length.

  Returns:
    A rank-1 Tensor containing the top `max_user_contribution` unique elements
    of the input `dataset`. If the total number of unique elements is less than
    or equal to `max_user_contribution`, returns the list of all unique
    elements.

  Raises:
    ValueError:
      -- If the shape of elements in `dataset` is not rank 1.
      -- If `max_user_contribution` is less than 1.
      -- If `string_max_bytes` is not `None` and is less than 1.
    TypeError: If `dataset.element_spec.dtype` must be `tf.string` is not
      `tf.string`.
  """
  top_elements, counts = get_top_elements_with_counts(
      dataset=dataset,
      max_user_contribution=max_user_contribution,
      string_max_bytes=string_max_bytes,
  )
  return tf.repeat(top_elements, counts)


class _TensorShapeNotFullyDefinedError(ValueError):
  pass


@tf.function
def to_stacked_tensor(ds: tf.data.Dataset) -> tf.Tensor:
  """Encodes the `tf.data.Dataset as stacked tensors.

  This is effectively the inverse of `tf.data.Dataset.from_tensor_slices()`.
  All elements from the input dataset are concatenated into a tensor structure,
  where the output structure matches the input `ds.element_spec`, and each
  output tensor will have the same shape plus one additional prefix dimension
  which elements are stacked in. For example, if the dataset contains  5
  elements with shape [3, 2], the returned tensor will have shape [5, 3, 2].
  Note that each element in the dataset could be as single tensor or a structure
  of tensors.

  Dataset elements must have fully-defined shapes. Any partially-defined element
  shapes will raise an error. If passing in a batched dataset, use
  `drop_remainder=True` to ensure the batched shape is fully defined.

  Args:
    ds: The input `tf.data.Dataset` to stack.

  Returns:
    A structure of tensors encoding the input dataset.

  Raises:
    ValueError: If any dataset element shape is not fully-defined.
  """
  py_typecheck.check_type(ds, tf.data.Dataset)

  def expanded_empty_tensor(tensor_spec: tf.TensorSpec) -> tf.Tensor:
    if not tensor_spec.shape.is_fully_defined():
      raise _TensorShapeNotFullyDefinedError()
    return tf.zeros(shape=(0,) + tensor_spec.shape, dtype=tensor_spec.dtype)

  with tf.name_scope('to_stacked_tensor'):
    try:
      initial_state = tf.nest.map_structure(
          expanded_empty_tensor, ds.element_spec
      )
    except _TensorShapeNotFullyDefinedError as shape_not_defined_error:
      raise ValueError(
          'Dataset elements must have fully-defined shapes. '
          f'Found: {ds.element_spec}'
      ) from shape_not_defined_error

  @tf.function
  def append_tensor(stacked: tf.Tensor, tensor: tf.Tensor) -> tf.Tensor:
    expanded_tensor = tf.expand_dims(tensor, axis=0)
    return tf.concat((stacked, expanded_tensor), axis=0)

  @tf.function
  def reduce_func(old_state, input_element):
    tf.nest.assert_same_structure(old_state, input_element)
    return tf.nest.map_structure(append_tensor, old_state, input_element)

  return ds.reduce(initial_state, reduce_func)
