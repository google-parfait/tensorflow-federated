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

import tensorflow as tf


def _apply_mask(y_true, sample_weight, masked_tokens):
  if sample_weight is None:
    sample_weight = tf.ones_like(y_true, tf.float32)
  for token in masked_tokens:
    mask = tf.cast(tf.not_equal(y_true, token), tf.float32)
    sample_weight = sample_weight * mask
  return sample_weight


class NumBatchesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of batches seen."""

  def __init__(self, name: str = 'num_batches', dtype=tf.int64):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(1)


class NumExamplesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of examples seen."""

  def __init__(self, name: str = 'num_examples', dtype=tf.int64):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(tf.shape(y_pred)[0])


class NumTokensCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts tokens seen after masking."""

  def __init__(self, masked_tokens=None, name='num_tokens', dtype=tf.int64):
    self._masked_tokens = masked_tokens or []
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    sample_weight = _apply_mask(y_true, sample_weight, self._masked_tokens)
    sample_weight = tf.reshape(sample_weight, [-1])
    super().update_state(sample_weight)


class MaskedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
  """An accuracy metric that masks some tokens."""

  def __init__(self, masked_tokens=None, name='accuracy', dtype=None):
    self._masked_tokens = masked_tokens or []
    super().__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    sample_weight = _apply_mask(y_true, sample_weight, self._masked_tokens)
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, num_classes])
    sample_weight = tf.reshape(sample_weight, [-1])
    super().update_state(y_true, y_pred, sample_weight)

  def get_config(self):
    config = super().get_config()
    config['masked_tokens'] = tuple(self._masked_tokens)
    return config
