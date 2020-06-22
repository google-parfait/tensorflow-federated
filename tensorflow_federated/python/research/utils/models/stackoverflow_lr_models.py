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
"""Sequence model functions for research baselines."""

import tensorflow as tf


def create_logistic_model(vocab_tokens_size, vocab_tags_size):
  """Logistic regression to predict tags of StackOverflow.

  Args:
      vocab_tokens_size: Size of token vocabulary to use.
      vocab_tags_size: Size of token vocabulary to use.

  Returns:
    `tf.keras.Model`.
  """
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          vocab_tags_size,
          activation='sigmoid',
          input_shape=(vocab_tokens_size,)
          ),
  ])

  return model
