# Copyright 2019, Google LLC.
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
"""Libraries for building Stack Overflow next-word prediction models."""

import tensorflow as tf


class TransposableEmbedding(tf.keras.layers.Layer):
  """A Keras layer implementing a transposed projection output layer."""

  def __init__(self, embedding_layer: tf.keras.layers.Embedding):
    super().__init__()
    self.embeddings = embedding_layer.embeddings

  # Placing `tf.matmul` under the `call` method is important for backpropagating
  # the gradients of `self.embeddings` in graph mode.
  def call(self, inputs):
    return tf.matmul(inputs, self.embeddings, transpose_b=True)


def create_recurrent_model(vocab_size: int,
                           embedding_size: int = 96,
                           num_lstm_layers: int = 1,
                           lstm_size: int = 670,
                           shared_embedding: bool = False) -> tf.keras.Model:
  """Constructs a recurrent model with an initial embeding layer.

  The resulting model embeds sequences of integer tokens (whose values vary
  between `0` and `vocab_size-1`) into an `embedding_size`-dimensional space.
  It then applies `num_lstm_layers` LSTM layers, each of size `lstm_size`.
  Each LSTM is followed by a dense layer mapping the output to `embedding_size`
  units. The model then has a final dense layer mapping to `vocab_size` logits
  units. Note that this model does not compute any kind of softmax on the final
  logits. This should instead be done in the loss function for the purposes of
  backpropagation.

  Args:
    vocab_size: Vocabulary size to use in the initial embedding layer.
    embedding_size: The size of the embedding layer.
    num_lstm_layers: The number of LSTM layers in the model.
    lstm_size: The size of each LSTM layer.
    shared_embedding: If set to `True`, the final layer of the model is a dense
      layer given by the transposition of the embedding layer. If `False`, the
      final dense layer is instead learned separately.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  if vocab_size < 1:
    raise ValueError('vocab_size must be a positive integer.')
  if embedding_size < 1:
    raise ValueError('embedding_size must be a positive integer.')
  if num_lstm_layers < 1:
    raise ValueError('num_lstm_layers must be a positive integer.')
  if lstm_size < 1:
    raise ValueError('lstm_size must be a positive integer.')

  inputs = tf.keras.layers.Input(shape=(None,))
  input_embedding = tf.keras.layers.Embedding(
      input_dim=vocab_size, output_dim=embedding_size, mask_zero=True)
  embedded = input_embedding(inputs)
  projected = embedded

  for _ in range(num_lstm_layers):
    layer = tf.keras.layers.LSTM(lstm_size, return_sequences=True)
    processed = layer(projected)
    projected = tf.keras.layers.Dense(embedding_size)(processed)

  if shared_embedding:
    transposed_embedding = TransposableEmbedding(input_embedding)
    logits = transposed_embedding(projected)
  else:
    logits = tf.keras.layers.Dense(vocab_size, activation=None)(projected)

  return tf.keras.Model(inputs=inputs, outputs=logits)
