# Copyright 2025, The TensorFlow Federated Authors.
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
"""Smoke test for TFF pip package: runs a federated computation.

This script is executed by test_python_package.sh inside a fresh venv with
only pip-installed packages. It exercises the C++ executor bindings by running
a simple federated computation using federated_map and federated_sum.
"""

import federated_language
import numpy as np
import tensorflow_federated as tff

NUM_CLIENTS = 5
EXPECTED_RESULT = 50


@tff.tensorflow.computation(np.int32)
def add_one(x):
  return x + 1


@tff.tensorflow.computation(np.int32)
def multiply_by_five(x):
  return x * 5


@federated_language.federated_computation(
    federated_language.FederatedType(np.int32, federated_language.CLIENTS)
)
def compute(client_values):
  # Each client adds 1: [1,1,1,1,1] -> [2,2,2,2,2]
  incremented = federated_language.federated_map(add_one, client_values)
  # Sum across clients: [2,2,2,2,2] -> 10
  summed = federated_language.federated_sum(incremented)
  # Multiply on server: 10 -> 50
  return federated_language.federated_map(multiply_by_five, summed)


def main():
  result = compute([1] * NUM_CLIENTS)
  assert result == EXPECTED_RESULT, f'Expected {EXPECTED_RESULT}, got {result}'
  print(f'Federated computation result: {result}')


if __name__ == '__main__':
  main()
