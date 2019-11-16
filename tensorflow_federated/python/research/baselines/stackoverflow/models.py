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
"""Sequence model functions for research baselines."""

import tensorflow.compat.v2 as tf


def create_recurrent_model(vocab_size,
                           embedding_size,
                           num_layers,
                           recurrent_layer_fn,
                           name='rnn'):
  """Constructs zero-padded keras model with the given parameters and cell.

  Args:
      vocab_size: Size of vocabulary.
      embedding_size: Size of embedding.
      num_layers: Number of LSTM layers to sequentially stack.
      recurrent_layer_fn: No-arg function which returns an instance of a
        subclass of `tf.keras.layers.RNN`, creating the cells of the recurrent
        model.
      name: (Optional) string to name the returned `tf.keras.Model`.

  Returns:
    `tf.keras.Model`.
  """
  inputs = tf.keras.layers.Input(shape=(None,))
  embedded = tf.keras.layers.Embedding(
      input_dim=vocab_size + 1,  # Add 1 for padding.
      output_dim=embedding_size,
      mask_zero=True)(
          inputs)
  projected = embedded

  for _ in range(num_layers):
    layer = recurrent_layer_fn()
    if not isinstance(layer, tf.keras.layers.RNN):
      raise ValueError('The `recurrent_layer_fn` parameter to '
                       '`create_recurrent_model` should return an instance of '
                       '`tf.keras.layers.Layer` which inherits from '
                       '`tf.keras.layers.RNN`; you passed a function returning '
                       '{}'.format(layer))
    processed = layer(projected)
    # A projection changes dimension from rnn_layer_size to input_embedding_size
    projected = tf.keras.layers.Dense(embedding_size)(processed)

  logits = tf.keras.layers.Dense(vocab_size + 1)(projected)

  return tf.keras.Model(inputs=inputs, outputs=logits, name=name)
