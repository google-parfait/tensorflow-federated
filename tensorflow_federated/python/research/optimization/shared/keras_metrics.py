# Lint as: python3
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
"""Libraries of Keras metrics."""

from typing import Any, Dict, List, Optional, TypeVar, Union

import tensorflow as tf


class NumBatchesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of batches seen."""

  def __init__(self, name: str = 'num_batches', dtype=tf.int64):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
    return super().update_state(1)


class NumExamplesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of examples seen."""

  def __init__(self, name: str = 'num_examples', dtype=tf.int64):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
    return super().update_state(tf.shape(y_pred)[0])


def _mask_ground_truth(
    y_true: tf.Tensor, sample_weight: tf.Tensor,
    token_to_mask: Optional[Union[tf.Tensor, str, float, int]]) -> tf.Tensor:
  """Generated sample weight to mask ground truth."""

  def _convert_masked_tokens_to_negative(y_true, token_to_mask) -> tf.Tensor:
    """Converts tokens to maks to negative values."""

    if token_to_mask is None:
      neq = tf.equal(y_true, y_true)
    else:
      neq = tf.not_equal(y_true, token_to_mask)
    mask = tf.cast(neq, y_true.dtype) * 2 - 1
    return mask

  if sample_weight is not None:
    sample_weight = tf.cast(
        tf.math.greater(
            _convert_masked_tokens_to_negative(y_true, token_to_mask), 0),
        tf.float32) * tf.cast(tf.reshape(sample_weight, [-1, 1]), tf.float32)
  else:
    sample_weight = tf.cast(
        tf.math.greater(
            _convert_masked_tokens_to_negative(y_true, token_to_mask), 0),
        tf.float32)
  return sample_weight


class FlattenedNumExamplesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of examples seen."""

  def __init__(self,
               name: str = 'num_flattened_examples',
               dtype=tf.int64,
               mask_zero=False):
    super().__init__(name, dtype)
    self._token_to_mask = None
    if mask_zero:
      self._token_to_mask = 0

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
    y_true = tf.reshape(y_true, [-1, 1])
    if sample_weight is not None:
      sample_weight = tf.reshape(sample_weight, [-1, 1])
    sample_weight = _mask_ground_truth(y_true, sample_weight,
                                       self._token_to_mask)
    return super().update_state(tf.reduce_sum(sample_weight))


maskable = TypeVar('maskable', tf.Tensor, str, float, int)


class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
  """An accuracy metric that flattens out sequences and masks a given token."""

  def __init__(self,
               vocab_size: int,
               name: str = 'accuracy',
               dtype=None,
               masked_tokens: Union[List[maskable], maskable] = None,
               mask_zero: bool = False):
    self._vocab_size = vocab_size
    self._tokens_to_mask = [masked_tokens] if (
        not isinstance(masked_tokens, List)) else masked_tokens  # type: List
    if mask_zero:
      self._tokens_to_mask.append(0)
    super().__init__(name, dtype=dtype)

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Flattens and masks `y_true`, `y_pred` and `sample_weight`.

    Args:
      y_true: Tensor representing ground truth of sequence model. Must contain
        only nonnegative indices.
      y_pred: Tensor representing per-element predictions over the sequence
        model's vocabulary; should have total number of elements equal to the
        total number of elements in `y_true` multiplied by `self._vocab_size`.
      sample_weight: (Optional) Tensor representing the per-element weights for
        computing accuracy over the sequence `y_true`. Must be broadcastable to
        the flattened shape of `y_true`.

    Returns:
      Update Op.
    """

    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.reshape(y_pred, [-1, self._vocab_size, 1])
    if sample_weight is not None:
      sample_weight = tf.reshape(sample_weight, [-1, 1])
    for token in self._tokens_to_mask:
      sample_weight = _mask_ground_truth(y_true, sample_weight, token)
    return super().update_state(y_true, y_pred, sample_weight)

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config['masked_tokens'] = tuple(self._tokens_to_mask)
    config['vocab_size'] = self._vocab_size
    return config
