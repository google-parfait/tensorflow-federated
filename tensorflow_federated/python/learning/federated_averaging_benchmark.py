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

import time

import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.learning import federated_averaging
from tensorflow_federated.python.learning import model_examples


class FederatedAveragingBenchmark(tf.test.Benchmark):
  """Inheriting TensorFlow's Benchmark capability."""

  def benchmark_simple_execution(self):
    num_clients = 10
    num_client_samples = 20
    batch_size = 4
    num_rounds = 10
    n_iters = 5

    ds = tf.data.Dataset.from_tensor_slices({
        "x": [[1., 2.]] * num_client_samples,
        "y": [[5.]] * num_client_samples
    }).batch(batch_size)

    federated_ds = [ds] * num_clients

    building_time_array = []
    for _ in range(n_iters):
      build_time_start = time.time()
      iterative_process = federated_averaging.build_federated_averaging_process(
          model_fn=model_examples.TrainableLinearRegression)
      build_time_stop = time.time()
      building_time_array.append(build_time_stop-build_time_start)
    self.report_benchmark(
        name="computation_building_time",
        wall_time=np.mean(building_time_array),
        iters=n_iters,
        extras={"std_dev": np.std(building_time_array)})

    initialization_array = []
    for _ in range(n_iters):
      initialization_start = time.time()
      server_state = iterative_process.initialize()
      initialization_stop = time.time()
      initialization_array.append(initialization_stop - initialization_start)
    self.report_benchmark(
        name="computation_initialization_time",
        wall_time=np.mean(initialization_array),
        iters=n_iters,
        extras={"std_dev": np.std(initialization_array)})

    next_state = server_state

    execution_array = []
    for _ in range(n_iters):
      rounds_start = time.time()
      next_state, _ = iterative_process.next(server_state, federated_ds)
      for _ in range(num_rounds - 1):
        next_state, _ = iterative_process.next(next_state, federated_ds)
      rounds_stop = time.time()
      execution_array.append(rounds_stop - rounds_start)
    self.report_benchmark(
        name="Time to execute {} rounds, {} clients, "
        "{} examples per client, batch size {}".format(
            num_rounds, num_clients, num_client_samples, batch_size),
        wall_time=np.mean(execution_array),
        iters=n_iters,
        extras={"std_dev": np.std(execution_array)})


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
