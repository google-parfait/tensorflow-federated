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
"""Benchmark for federated_averaging."""

import collections
import time

import numpy as np
import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import federated_averaging
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model_examples

tf.compat.v1.enable_v2_behavior()

BATCH_SIZE = 100


def wrap_data(images, digits):
  output_sequence = []
  for _ in range(10):
    output_sequence.append(
        collections.OrderedDict(
            x=np.array(images, dtype=np.float32),
            y=np.array(digits, dtype=np.int32)))
  return output_sequence


def generate_fake_mnist_data():
  fake_x_data = np.random.random_sample((100, 784))
  fake_y_data = np.random.choice([k for k in range(10)], (100,))
  return [wrap_data(fake_x_data, fake_y_data) for k in range(10)]


def executors_benchmark(fn):
  """Generates different local executors for basic benchmarks."""

  def wrapped_fn(self):
    """Runs `fn` against different local executor stacks."""
    # TODO(b/148233458): Re-enable reference executor benchmarks when possible.
    tff.framework.set_default_executor()
    fn(self, "local executor")
    tff.framework.set_default_executor(tff.framework.sizing_executor_factory())
    fn(self, "sizing executor")
    tff.framework.set_default_executor(
        tff.framework.local_executor_factory(clients_per_thread=2))
    fn(self, "local executor, 2 clients per worker")
    tff.framework.set_default_executor(
        tff.framework.local_executor_factory(clients_per_thread=4))
    fn(self, "local executor, 4 clients per worker")
    tff.framework.set_default_executor()

  return wrapped_fn


class FederatedAveragingBenchmark(tf.test.Benchmark):
  """Inheriting TensorFlow's Benchmark capability."""

  @executors_benchmark
  def benchmark_simple_execution(self, executor_id):
    num_clients = 10
    num_client_samples = 20
    batch_size = 4
    num_rounds = 10

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1., 2.]] * num_client_samples,
            y=[[5.]] * num_client_samples)).batch(batch_size)

    federated_ds = [ds] * num_clients

    building_time_array = []
    build_time_start = time.time()
    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.TrainableLinearRegression)
    build_time_stop = time.time()
    building_time_array.append(build_time_stop - build_time_start)
    self.report_benchmark(
        name="computation_building_time, simple execution "
        "TrainableLinearRegression, executor {}".format(executor_id),
        wall_time=np.mean(building_time_array),
        iters=1)

    initialization_array = []
    initialization_start = time.time()
    initial_state = iterative_process.initialize()
    initialization_stop = time.time()
    initialization_array.append(initialization_stop - initialization_start)
    self.report_benchmark(
        name="computation_initialization_time, simple execution "
        "TrainableLinearRegression, executor {}".format(executor_id),
        wall_time=np.mean(initialization_array),
        iters=1)

    next_state = initial_state

    execution_array = []
    for _ in range(num_rounds - 1):
      round_start = time.time()
      next_state, _ = iterative_process.next(next_state, federated_ds)
      round_stop = time.time()
      execution_array.append(round_stop - round_start)
    self.report_benchmark(
        name="Average per round time, {} clients, "
        "{} examples per client, batch size {}, "
        "TrainableLinearRegression, executor {}".format(num_clients,
                                                        num_client_samples,
                                                        batch_size,
                                                        executor_id),
        wall_time=np.mean(execution_array),
        iters=num_rounds,
        extras={"std_dev": np.std(execution_array)})

  @executors_benchmark
  def benchmark_learning_keras_model_mnist(self, executor_id):
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
      return keras_utils.from_keras_model(
          model,
          dummy_batch=federated_train_data[0][0],
          loss=tf.keras.losses.SparseCategoricalCrossentropy())

    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
    computation_building_stop = time.time()
    building_time = computation_building_stop - computation_building_start
    self.report_benchmark(
        name="computation_building_time, "
        "tff.learning Keras model, executor {}".format(executor_id),
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
        "tff.learning Keras model, executor {}".format(executor_id),
        wall_time=np.mean(execution_array),
        iters=n_rounds,
        extras={"std_dev": np.std(execution_array)})


if __name__ == "__main__":
  test.main()
