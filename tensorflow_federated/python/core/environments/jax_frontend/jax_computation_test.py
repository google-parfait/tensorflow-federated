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
import federated_language
import jax
import ml_dtypes
import numpy as np

from tensorflow_federated.python.core.environments.jax_frontend import jax_computation


class ToNumpyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar', lambda: jax.numpy.array(1), 1),
      ('array', lambda: jax.numpy.array([1, 2, 3]), np.array([1, 2, 3])),
      (
          'list_array',
          lambda: [jax.numpy.array(1), jax.numpy.array(2), jax.numpy.array(3)],
          [1, 2, 3],
      ),
  )
  def test_returns_expected_result(self, value_factory, expected_result):
    value = value_factory()
    actual_result = jax_computation._to_numpy(value)

    if isinstance(actual_result, (np.ndarray, np.generic)):
      np.testing.assert_array_equal(actual_result, expected_result)
    else:
      self.assertEqual(actual_result, expected_result)


class JaxComputationTest(parameterized.TestCase):

  def test_returns_concrete_computation_with_no_arg(self):
    @jax_computation.jax_computation()
    def _comp():
      return 1

    self.assertIsInstance(
        _comp, federated_language.framework.ConcreteComputation
    )
    expected_type = federated_language.FunctionType(None, np.int32)
    self.assertEqual(_comp.type_signature, expected_type)

  def test_returns_concrete_computation_with_one_arg(self):
    @jax_computation.jax_computation(np.int32)
    def _comp(x):
      return jax.numpy.add(x, 1)

    self.assertIsInstance(
        _comp, federated_language.framework.ConcreteComputation
    )
    expected_type = federated_language.FunctionType(np.int32, np.int32)
    self.assertEqual(_comp.type_signature, expected_type)

  def test_returns_concrete_computation_with_two_args(self):
    @jax_computation.jax_computation(np.int32, np.int32)
    def _comp(x, y):
      return jax.numpy.add(x, y)

    self.assertIsInstance(
        _comp, federated_language.framework.ConcreteComputation
    )
    expected_type = federated_language.FunctionType(
        federated_language.StructWithPythonType(
            [('x', np.int32), ('y', np.int32)], collections.OrderedDict
        ),
        np.int32,
    )
    self.assertEqual(_comp.type_signature, expected_type)

  def test_returns_concrete_computation_with_correct_arg_order(self):

    @jax_computation.jax_computation(
        federated_language.TensorType(np.int32, (10,)), np.int32
    )
    def _comp(y, x):
      return jax.numpy.add(x, jax.numpy.sum(y))

    self.assertIsInstance(
        _comp, federated_language.framework.ConcreteComputation
    )
    expected_type = federated_language.FunctionType(
        federated_language.StructWithPythonType(
            [
                ('y', federated_language.TensorType(np.int32, (10,))),
                ('x', np.int32),
            ],
            collections.OrderedDict,
        ),
        np.int32,
    )
    self.assertEqual(_comp.type_signature, expected_type)

  @parameterized.named_parameters(
      ('bool', federated_language.TensorType(np.bool_)),
      ('int8', federated_language.TensorType(np.int8)),
      ('int16', federated_language.TensorType(np.int16)),
      ('int32', federated_language.TensorType(np.int32)),
      ('uint8', federated_language.TensorType(np.uint8)),
      ('uint16', federated_language.TensorType(np.uint16)),
      ('uint32', federated_language.TensorType(np.uint32)),
      ('float16', federated_language.TensorType(np.float16)),
      ('float32', federated_language.TensorType(np.float32)),
      ('complex64', federated_language.TensorType(np.complex64)),
      ('bfloat16', federated_language.TensorType(ml_dtypes.bfloat16)),
      ('generic', federated_language.TensorType(np.int32)),
      ('array', federated_language.TensorType(np.int32, shape=[3])),
  )
  def test_returns_concrete_computation_with_dtype(self, type_spec):
    @jax_computation.jax_computation(type_spec)
    def _comp(x):
      return x

    self.assertIsInstance(
        _comp, federated_language.framework.ConcreteComputation
    )
    expected_type = federated_language.FunctionType(type_spec, type_spec)
    self.assertEqual(_comp.type_signature, expected_type)

  @parameterized.named_parameters(
      ('int64', federated_language.TensorType(np.int64)),
      ('uint64', federated_language.TensorType(np.uint64)),
      ('float64', federated_language.TensorType(np.float64)),
      ('complex128', federated_language.TensorType(np.complex128)),
  )
  def test_returns_concrete_computation_with_dtype_and_enable_x64(
      self, type_spec
  ):
    jax.config.update('jax_enable_x64', True)

    @jax_computation.jax_computation(type_spec)
    def _comp(x):
      return x

    self.assertIsInstance(
        _comp, federated_language.framework.ConcreteComputation
    )
    expected_type = federated_language.FunctionType(type_spec, type_spec)
    self.assertEqual(_comp.type_signature, expected_type)
    jax.config.update('jax_enable_x64', False)

  @parameterized.named_parameters(
      ('int64', federated_language.TensorType(np.int64)),
      ('uint64', federated_language.TensorType(np.uint64)),
      ('float64', federated_language.TensorType(np.float64)),
      ('complex128', federated_language.TensorType(np.complex128)),
      ('str', federated_language.TensorType(np.str_)),
  )
  def test_raises_raises_value_error_with_dtype(self, type_spec):
    with self.assertRaises(ValueError):

      @jax_computation.jax_computation(type_spec)
      def _comp(x):
        return x, x

  def test_returns_polymorphic_computation(self):
    @jax_computation.jax_computation()
    def _comp(x):
      return jax.numpy.add(x, 1)

    self.assertIsInstance(
        _comp, federated_language.framework.PolymorphicComputation
    )


if __name__ == '__main__':
  absltest.main()
