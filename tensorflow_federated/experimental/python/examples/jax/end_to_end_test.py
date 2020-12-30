# Copyright 2020, The TensorFlow Federated Authors.
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

from absl.testing import absltest
import jax
import numpy as np
import tensorflow as tf

from tensorflow_federated.experimental.python.core.api import computations
from tensorflow_federated.experimental.python.core.backends.xla import executor
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_test_utils


class _XlaExecutorFactoryForTesting(executor_factory.ExecutorFactory):

  def create_executor(self, cardinalities):
    return executor.XlaExecutor()

  def clean_up_executors(self):
    pass


class EndToEndTest(absltest.TestCase):

  # TODO(b/175888145): Extend and clean this up as the implementation of JAX
  # and XLA support gets more complete.

  def test_add_numbers(self):

    @computations.jax_computation(tf.int32, tf.int32)
    def foo(x, y):
      return jax.numpy.add(x, y)

    with executor_test_utils.install_executor(_XlaExecutorFactoryForTesting()):
      result = foo(np.int32(20), np.int32(30))

    self.assertEqual(result, 50)


if __name__ == '__main__':
  absltest.main()
