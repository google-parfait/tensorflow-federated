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

import collections

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np

from tensorflow_federated.python.core.environments.jax_frontend import jax_computation
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import polymorphic_computation
from tensorflow_federated.python.core.impl.types import computation_types


class JaxComputationTest(parameterized.TestCase):

  def test_returns_concrete_computation_with_no_arg(self):
    @jax_computation.jax_computation()
    def _comp():
      return 1

    self.assertIsInstance(_comp, computation_impl.ConcreteComputation)
    expected_type = computation_types.FunctionType(None, np.int32)
    self.assertEqual(_comp.type_signature, expected_type)

  def test_returns_concrete_computation_with_one_arg(self):
    @jax_computation.jax_computation(np.int32)
    def _comp(x):
      return jax.numpy.add(x, 1)

    self.assertIsInstance(_comp, computation_impl.ConcreteComputation)
    expected_type = computation_types.FunctionType(np.int32, np.int32)
    self.assertEqual(_comp.type_signature, expected_type)

  def test_returns_concrete_computation_with_two_args(self):
    @jax_computation.jax_computation(np.int32, np.int32)
    def _comp(x, y):
      return jax.numpy.add(x, y)

    self.assertIsInstance(_comp, computation_impl.ConcreteComputation)
    expected_type = computation_types.FunctionType(
        computation_types.StructWithPythonType(
            [('x', np.int32), ('y', np.int32)], collections.OrderedDict
        ),
        np.int32,
    )
    self.assertEqual(_comp.type_signature, expected_type)

  def test_returns_concrete_computation_with_correct_arg_order(self):
    @jax_computation.jax_computation(
        computation_types.TensorType(np.int32, (10,)), np.int32
    )
    def _comp(y, x):
      return jax.numpy.add(x, jax.numpy.sum(y))

    self.assertIsInstance(_comp, computation_impl.ConcreteComputation)
    expected_type = computation_types.FunctionType(
        computation_types.StructWithPythonType(
            [
                ('y', computation_types.TensorType(np.int32, (10,))),
                ('x', np.int32),
            ],
            collections.OrderedDict,
        ),
        np.int32,
    )
    self.assertEqual(_comp.type_signature, expected_type)

  @parameterized.named_parameters(
      ('bool', computation_types.TensorType(np.bool_)),
      ('int8', computation_types.TensorType(np.int8)),
      ('int16', computation_types.TensorType(np.int16)),
      ('int32', computation_types.TensorType(np.int32)),
      ('uint8', computation_types.TensorType(np.uint8)),
      ('uint16', computation_types.TensorType(np.uint16)),
      ('uint32', computation_types.TensorType(np.uint32)),
      ('float16', computation_types.TensorType(np.float16)),
      ('float32', computation_types.TensorType(np.float32)),
      ('complex64', computation_types.TensorType(np.complex64)),
      ('generic', computation_types.TensorType(np.int32)),
      ('array', computation_types.TensorType(np.int32, shape=[3])),
  )
  def test_returns_concrete_computation_with_dtype(self, type_spec):
    @jax_computation.jax_computation(type_spec)
    def _comp(x):
      return x

    self.assertIsInstance(_comp, computation_impl.ConcreteComputation)
    expected_type = computation_types.FunctionType(type_spec, type_spec)
    self.assertEqual(_comp.type_signature, expected_type)

  @parameterized.named_parameters(
      ('int64', computation_types.TensorType(np.int64)),
      ('uint64', computation_types.TensorType(np.uint64)),
      ('float64', computation_types.TensorType(np.float64)),
      ('complex128', computation_types.TensorType(np.complex128)),
  )
  def test_returns_concrete_computation_with_dtype_and_enable_x64(
      self, type_spec
  ):
    jax.config.update('jax_enable_x64', True)

    @jax_computation.jax_computation(type_spec)
    def _comp(x):
      return x

    self.assertIsInstance(_comp, computation_impl.ConcreteComputation)
    expected_type = computation_types.FunctionType(type_spec, type_spec)
    self.assertEqual(_comp.type_signature, expected_type)
    jax.config.update('jax_enable_x64', False)

  @parameterized.named_parameters(
      ('int64', computation_types.TensorType(np.int64)),
      ('uint64', computation_types.TensorType(np.uint64)),
      ('float64', computation_types.TensorType(np.float64)),
      ('complex128', computation_types.TensorType(np.complex128)),
      ('str', computation_types.TensorType(np.str_)),
  )
  async def test_raises_raises_value_error_with_dtype(self, type_spec):
    with self.assertRaises(ValueError):

      @jax_computation.jax_computation(type_spec)
      def _comp(x):
        return x, x

  def test_returns_polymorphic_computation(self):
    @jax_computation.jax_computation()
    def _comp(x):
      return jax.numpy.add(x, 1)

    self.assertIsInstance(_comp, polymorphic_computation.PolymorphicComputation)


if __name__ == '__main__':
  absltest.main()
