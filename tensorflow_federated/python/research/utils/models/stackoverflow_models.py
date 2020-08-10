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


class TransposableEmbedding(tf.keras.layers.Embedding):
  """A Keras Embedding layer implements a transposed projection for output."""

  def reverse_project(self, inputs):
    return tf.matmul(inputs, self.embeddings, transpose_b=True)


def create_recurrent_model(vocab_size=10000,
                           num_oov_buckets=1,
                           embedding_size=96,
                           latent_size=670,
                           num_layers=1,
                           name='rnn',
                           shared_embedding=False):
  """Constructs zero-padded keras model with the given parameters and cell.

  Args:
      vocab_size: Size of vocabulary to use.
      num_oov_buckets: Number of out of vocabulary buckets.
      embedding_size: The size of the embedding.
      latent_size: The size of the recurrent state.
      num_layers: The number of layers.
      name: (Optional) string to name the returned `tf.keras.Model`.
      shared_embedding: (Optional) Whether to tie the input and output
        embeddings.

  Returns:
    `tf.keras.Model`.
  """
  extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
  inputs = tf.keras.layers.Input(shape=(None,))
  input_embedding = TransposableEmbedding(
      input_dim=extended_vocab_size, output_dim=embedding_size, mask_zero=True)
  embedded = input_embedding(inputs)
  projected = embedded

  for _ in range(num_layers):
    layer = tf.keras.layers.LSTM(latent_size, return_sequences=True)
    processed = layer(projected)
    # A projection changes dimension from rnn_layer_size to input_embedding_size
    projected = tf.keras.layers.Dense(embedding_size)(processed)

  if shared_embedding:
    logits = input_embedding.reverse_project(projected)
  else:
    logits = tf.keras.layers.Dense(
        extended_vocab_size, activation=None)(
            projected)

  return tf.keras.Model(inputs=inputs, outputs=logits, name=name)
