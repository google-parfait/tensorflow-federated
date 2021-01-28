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

from tensorflow_federated.experimental.python.core.impl.wrappers import computation_wrapper_instances
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.computation import computation_impl


class JaxWrapperTest(absltest.TestCase):

  def test_invoke_with_single_arg_fn(self):

    @computation_wrapper_instances.jax_wrapper(np.int32)
    def foo(x):
      return jax.numpy.add(x, np.int32(10))

    self.assertIsInstance(foo, computation_impl.ComputationImpl)
    self.assertEqual(str(foo.type_signature), '(int32 -> int32)')

  def test_invoke_with_two_arg_fn(self):

    @computation_wrapper_instances.jax_wrapper(np.int32, np.int32)
    def foo(x, y):
      return jax.numpy.add(x, y)

    self.assertIsInstance(foo, computation_impl.ComputationImpl)
    self.assertEqual(str(foo.type_signature), '(<x=int32,y=int32> -> int32)')

  def test_arg_ordering(self):

    @computation_wrapper_instances.jax_wrapper(
        computation_types.TensorType(np.int32, 10), np.int32)
    def foo(b, a):
      return jax.numpy.add(a, jax.numpy.sum(b))

    self.assertIsInstance(foo, computation_impl.ComputationImpl)
    self.assertEqual(
        str(foo.type_signature), '(<b=int32[10],a=int32> -> int32)')


if __name__ == '__main__':
  absltest.main()
