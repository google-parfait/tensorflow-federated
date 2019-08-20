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
"""Benchmark for learning.federated_averaging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time

import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import federated_averaging
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model_examples

BATCH_SIZE = 100


def wrap_data(images, digits):
  output_sequence = []
  for _ in range(10):
    output_sequence.append(
        collections.OrderedDict([("x", np.array(images, dtype=np.float32)),
                                 ("y", np.array(digits, dtype=np.int32))]))
  return output_sequence


def generate_fake_mnist_data():
  fake_x_data = np.random.random_sample((100, 784))
  fake_y_data = np.random.choice([k for k in range(10)], (100,))
  return [wrap_data(fake_x_data, fake_y_data) for k in range(10)]


class FederatedAveragingBenchmark(tf.test.Benchmark):
  """Inheriting TensorFlow's Benchmark capability."""

  def benchmark_simple_execution(self):
    num_clients = 10
    num_client_samples = 20
    batch_size = 4
    num_rounds = 10

    ds = tf.data.Dataset.from_tensor_slices({
        "x": [[1., 2.]] * num_client_samples,
        "y": [[5.]] * num_client_samples
    }).batch(batch_size)

    federated_ds = [ds] * num_clients

    building_time_array = []
    build_time_start = time.time()
    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.TrainableLinearRegression)
    build_time_stop = time.time()
    building_time_array.append(build_time_stop - build_time_start)
    self.report_benchmark(
        name="computation_building_time, simple execution "
        "TrainableLinearRegression",
        wall_time=np.mean(building_time_array),
        iters=1)

    initialization_array = []
    initialization_start = time.time()
    server_state = iterative_process.initialize()
    initialization_stop = time.time()
    initialization_array.append(initialization_stop - initialization_start)
    self.report_benchmark(
        name="computation_initialization_time, simple execution "
        "TrainableLinearRegression",
        wall_time=np.mean(initialization_array),
        iters=1)

    next_state = server_state

    execution_array = []
    next_state, _ = iterative_process.next(server_state, federated_ds)
    for _ in range(num_rounds - 1):
      round_start = time.time()
      next_state, _ = iterative_process.next(next_state, federated_ds)
      round_stop = time.time()
      execution_array.append(round_stop - round_start)
    self.report_benchmark(
        name="Time to execute {} rounds, {} clients, "
        "{} examples per client, batch size {}, "
        "TrainableLinearRegression".format(num_rounds, num_clients,
                                           num_client_samples, batch_size),
        wall_time=np.mean(execution_array),
        iters=num_rounds,
        extras={"std_dev": np.std(execution_array)})

  def benchmark_fc_api_mnist(self):
    """Code adapted from FC API tutorial ipynb."""
    n_rounds = 10

    batch_type = tff.NamedTupleType([("x",
                                      tff.TensorType(tf.float32, [None, 784])),
                                     ("y", tff.TensorType(tf.int32, [None]))])

    model_type = tff.NamedTupleType([("weights",
                                      tff.TensorType(tf.float32, [784, 10])),
                                     ("bias", tff.TensorType(tf.float32,
                                                             [10]))])

    local_data_type = tff.SequenceType(batch_type)

    server_model_type = tff.FederatedType(model_type, tff.SERVER)
    client_data_type = tff.FederatedType(local_data_type, tff.CLIENTS)
    server_float_type = tff.FederatedType(tf.float32, tff.SERVER)

    computation_building_start = time.time()

    # pylint: disable=missing-docstring
    @tff.tf_computation(model_type, batch_type)
    def batch_loss(model, batch):
      predicted_y = tf.nn.softmax(
          tf.matmul(batch.x, model.weights) + model.bias)
      return -tf.reduce_mean(
          tf.reduce_sum(
              tf.one_hot(batch.y, 10) * tf.math.log(predicted_y), axis=[1]))

    initial_model = {
        "weights": np.zeros([784, 10], dtype=np.float32),
        "bias": np.zeros([10], dtype=np.float32)
    }

    @tff.tf_computation(model_type, batch_type, tf.float32)
    def batch_train(initial_model, batch, learning_rate):
      model_vars = tff.utils.create_variables("v", model_type)
      init_model = tff.utils.assign(model_vars, initial_model)

      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
      with tf.control_dependencies([init_model]):
        train_model = optimizer.minimize(batch_loss(model_vars, batch))

      with tf.control_dependencies([train_model]):
        return tff.utils.identity(model_vars)

    @tff.federated_computation(model_type, tf.float32, local_data_type)
    def local_train(initial_model, learning_rate, all_batches):

      @tff.federated_computation(model_type, batch_type)
      def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)

      return tff.sequence_reduce(all_batches, initial_model, batch_fn)

    @tff.federated_computation(server_model_type, server_float_type,
                               client_data_type)
    def federated_train(model, learning_rate, data):
      return tff.federated_mean(
          tff.federated_map(local_train, [
              tff.federated_broadcast(model),
              tff.federated_broadcast(learning_rate), data
          ]))

    computation_building_stop = time.time()
    building_time = computation_building_stop - computation_building_start
    self.report_benchmark(
        name="computation_building_time, FC API",
        wall_time=building_time,
        iters=1)

    model = initial_model
    learning_rate = 0.1

    federated_data = generate_fake_mnist_data()

    execution_array = []
    for _ in range(n_rounds):
      execution_start = time.time()
      model = federated_train(model, learning_rate, federated_data)
      execution_stop = time.time()
      execution_array.append(execution_stop - execution_start)

    self.report_benchmark(
        name="Average per round execution time, FC API",
        wall_time=np.mean(execution_array),
        iters=n_rounds,
        extras={"std_dev": np.std(execution_array)})

  def benchmark_learning_keras_model_mnist(self):
    """Code adapted from MNIST learning tutorial ipynb."""
    federated_train_data = generate_fake_mnist_data()
    n_rounds = 10
    computation_building_start = time.time()

    # pylint: disable=missing-docstring
    def model_fn():
      model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(784,)),
          tf.keras.layers.Dense(
              10,
              kernel_initializer="zeros",
              bias_initializer="zeros",
              activation=tf.nn.softmax)
      ])

      model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          optimizer=tf.keras.optimizers.SGD(0.1))

      return keras_utils.from_compiled_keras_model(model,
                                                   federated_train_data[0][0])

    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn)
    computation_building_stop = time.time()
    building_time = computation_building_stop - computation_building_start
    self.report_benchmark(
        name="computation_building_time, "
        "tff.learning Keras model",
        wall_time=building_time,
        iters=1)

    state = iterative_process.initialize()

    execution_array = []
    for _ in range(n_rounds):
      execution_start = time.time()
      state, _ = iterative_process.next(state, federated_train_data)
      execution_stop = time.time()
      execution_array.append(execution_stop - execution_start)

    self.report_benchmark(
        name="Average per round execution time, "
        "tff.learning Keras model",
        wall_time=np.mean(execution_array),
        iters=n_rounds,
        extras={"std_dev": np.std(execution_array)})


if __name__ == "__main__":
  test.main()
