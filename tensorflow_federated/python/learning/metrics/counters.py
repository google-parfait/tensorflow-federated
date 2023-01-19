# Copyright 2022, The TensorFlow Federated Authors.
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
"""A library of `tf.keras.metrics.Metrics` for learning."""

import numpy as np
import tensorflow as tf


class NumExamplesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of examples seen.

  This metric expects `np.ndarray` or `tf.Tensor` values. To work with
  multi-output models it will raise an error if the inputs are Python structures
  of Python `int` or `float` values. Please use `tf.convert_to_tensor` or
  `np.asarray`.

  Note: The number of examples is computed as the size of the first dimension of
  the labels. If the batch dimension is not the first dimension, or there are
  multiple labels per example, then this metric may be unsuitable.

  IMPORTANT: This metric ignores sample weighting, counting each example
  uniformly.
  """

  def __init__(self, name='num_examples', dtype=tf.int64):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred=None, sample_weight=None):
    del y_pred  # Unused
    # In case we have multiple labels, we use the first dimension of the first
    # label to compute the batch size.
    labels = tf.nest.flatten(y_true)
    if not all(tf.is_tensor(l) or isinstance(l, np.ndarray) for l in labels):
      raise ValueError(
          'NumExamplesCounter only works with `numpy.ndarray` or '
          '`tensorflow.Tensor` types. Received a structure with '
          'other values; consider using `np.asarray` or '
          f'`tf.convert_to_tensor`. Got: {labels}'
      )
    return super().update_state(tf.shape(labels[0])[0], sample_weight=None)


class NumBatchesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of batches seen.

  NOTE: This metric ignores sample weighting, counting each batch uniformly.
  """

  def __init__(self, name='num_batches', dtype=tf.int64):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred=None, sample_weight=None):
    del y_true  # Unused.
    del y_pred  # Unused.
    return super().update_state(1, sample_weight=None)
