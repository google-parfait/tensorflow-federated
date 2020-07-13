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
"""Libraries to prepare Shakespeare Keras models for CharRNN experiments."""

import functools

import tensorflow as tf


def create_recurrent_model(vocab_size: int,
                           sequence_length: int,
                           mask_zero: bool = True) -> tf.keras.Model:
  """Creates a RNN model using LSTM layers for Shakespeare language models.

  This replicates the model structure in the paper:

  Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera
    y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629

  Args:
    vocab_size: the size of the vocabulary, used as a dimension in the input
      embedding.
    sequence_length: the length of input sequences.
    mask_zero: Whether to mask zero tokens in the input.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  model = tf.keras.Sequential()
  model.add(
      tf.keras.layers.Embedding(
          input_dim=vocab_size,
          input_length=sequence_length,
          output_dim=8,
          mask_zero=mask_zero))
  lstm_layer_builder = functools.partial(
      tf.keras.layers.LSTM,
      units=256,
      kernel_initializer='he_normal',
      return_sequences=True,
      stateful=False)
  model.add(lstm_layer_builder())
  model.add(lstm_layer_builder())
  model.add(tf.keras.layers.Dense(vocab_size))  # Note: logits, no softmax.
  return model
