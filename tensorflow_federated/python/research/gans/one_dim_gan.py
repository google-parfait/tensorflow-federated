# Lint as: python3
# Copyright 2018, The TensorFlow Federated Authors.
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
"""A trivial one-dimensional problem, intended for use in testing."""

import tensorflow as tf

layers = tf.keras.layers

BATCH_SIZE = 8
NOISE_DIM = 8


def create_real_data(batch_size=BATCH_SIZE):
  """Generates batches of scalars from a mixture of Guassians."""

  @tf.function
  def gen_sample():
    # Mixture of normal distributions, each equally likely:
    logits = tf.constant([[1.0, 1.0]])
    means = tf.constant([-1.0, 2.0])
    stddevs = tf.constant([0.01, 0.01])
    i = tf.random.categorical(logits, 1)
    i = tf.reshape(i, ())
    return tf.random.normal(shape=(1,), mean=means[i], stddev=stddevs[i])

  return (tf.data.Dataset.from_tensors(0).repeat().map(
      lambda _: gen_sample()).batch(batch_size))


def create_generator_inputs(batch_size=BATCH_SIZE, noise_dim=NOISE_DIM):
  return tf.data.Dataset.from_tensors(0).repeat().map(
      lambda _: tf.random.normal([batch_size, noise_dim]))


def create_generator(noise_dim=NOISE_DIM):
  return tf.keras.Sequential([
      layers.Input(shape=[noise_dim]),
      layers.Dense(16, activation='relu'),
      layers.BatchNormalization(momentum=0.999, epsilon=0.001),
      layers.Dense(1, activation='linear')
  ],
                             name='one_dim_generator')


def create_discriminator():
  return tf.keras.Sequential([
      layers.Input(shape=[1]),
      layers.Dense(16, activation='relu'),
      layers.BatchNormalization(momentum=0.999, epsilon=0.001),
      layers.Dense(1, activation='linear')
  ],
                             name='one_dim_discriminator')
