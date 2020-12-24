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
import tensorflow as tf
from tensorflow_federated.experimental.python.core.impl.jax_context import jax_computation_context
from tensorflow_federated.python.core.api import computations


class JaxComputationContextTest(absltest.TestCase):

  def test_ingest_raises_not_implemented_error(self):

    context = jax_computation_context.JaxComputationContext()
    with self.assertRaisesRegex(NotImplementedError,
                                'JAX code is not currently supported'):
      context.ingest(10, tf.int32)

  def test_invoke_raises_not_implemented_error(self):

    @computations.tf_computation
    def foo():
      return tf.constant(10)

    context = jax_computation_context.JaxComputationContext()
    with self.assertRaisesRegex(NotImplementedError,
                                'JAX code is not currently supported'):
      context.invoke(foo, None)


if __name__ == '__main__':
  absltest.main()
