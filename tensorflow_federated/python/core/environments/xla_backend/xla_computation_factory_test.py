# Copyright 2021, The TensorFlow Federated Authors.
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

from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.lib import xla_client
import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.xla_backend import runtime
from tensorflow_federated.python.core.environments.xla_backend import xla_computation_factory
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def _to_python(obj):
  if isinstance(obj, np.ndarray):
    return obj.tolist()
  else:
    return obj


def _run_jax(
    computation_proto: pb.Computation,
    type_spec: computation_types.Type,
    arg: Optional[object] = None,
) -> object:
  backend = jax.lib.xla_bridge.get_backend()
  comp = runtime.ComputationCallable(computation_proto, type_spec, backend)
  if arg is not None:
    result = comp(arg)
  else:
    result = comp()
  return result


class CreateConstantTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'int_and_tensor_type_scalar',
          1,
          computation_types.TensorType(np.int32),
          np.int32(1),
      ),
      (
          'float_and_tensor_type_scalar',
          1.0,
          computation_types.TensorType(np.float32),
          np.float32(1.0),
      ),
      (
          'int_and_tensor_type_array',
          1,
          computation_types.TensorType(np.int32, [3]),
          np.array([1] * 3, np.int32),
      ),
      (
          'int_and_struct_type_unnamed',
          1,
          computation_types.StructType([np.int32] * 3),
          structure.Struct([(None, 1)] * 3),
      ),
      (
          'int_and_struct_type_named',
          1,
          computation_types.StructType([
              ('a', np.int32),
              ('b', np.int32),
              ('c', np.int32),
          ]),
          structure.Struct([('a', 1), ('b', 1), ('c', 1)]),
      ),
      (
          'int_and_struct_type_nested',
          1,
          computation_types.StructType([[np.int32] * 3] * 3),
          structure.Struct([(None, structure.Struct([(None, 1)] * 3))] * 3),
      ),
      (
          'tuple_empty',
          (),
          computation_types.StructType([]),
          structure.Struct([]),
      ),
      # (
      #     'tuple_and_struct_type_unnamed',
      #     (1, 2, 3.0),
      #     computation_types.StructType([np.int32, np.int32, np.float32]),
      #     structure.Struct([(None, 1), (None, 2), (None, 3.0)]),
      # ),
      # (
      #     'tuple_and_struct_type_named',
      #     (1, 2, 3.0),
      #     computation_types.StructType([
      #         ('a', np.int32),
      #         ('b', np.int32),
      #         ('c', np.float32),
      #     ]),
      #     structure.Struct([('a', 1), ('b', 2), ('c', 3.0)]),
      # ),
      # (
      #     'tuple_nested_and_struct_type_nested',
      #     (1, (2, 3.0)),
      #     computation_types.StructType([
      #         np.int32,
      #         computation_types.StructType([np.int32, np.float32]),
      #     ]),
      #     structure.Struct([
      #         (None, 1),
      #         (None, structure.Struct([(None, 2), (None, 3.0)])),
      #     ]),
      # ),
      # (
      #     'dict_and_struct_type_named',
      #     {'a': 1, 'b': 2, 'c': 3.0},
      #     computation_types.StructType([
      #         ('a', np.int32),
      #         ('b', np.int32),
      #         ('c', np.float32),
      #     ]),
      #     structure.Struct([('a', 1), ('b', 2), ('c', 3.0)]),
      # ),
      # (
      #     'dict_nested_and_struct_type_nested',
      #     {'a': 1, 'b': {'c': 2, 'd': 3.0}},
      #     computation_types.StructType([
      #         ('a', np.int32),
      #         (
      #             'b',
      #             computation_types.StructType([
      #                 ('c', np.int32),
      #                 ('d', np.float32),
      #             ]),
      #         ),
      #     ]),
      #     structure.Struct([
      #         ('a', 1),
      #         ('b', structure.Struct([('c', 2), ('d', 3.0)])),
      #     ]),
      # ),
  )
  def test_returns_result(self, value, type_spec, expected_result):
    factory = xla_computation_factory.XlaComputationFactory()

    proto, actual_type = factory.create_constant(value, type_spec)

    actual_result = _run_jax(proto, actual_type)
    self.assertIsInstance(actual_result, type(expected_result))
    actual_result = _to_python(actual_result)
    expected_result = _to_python(expected_result)
    self.assertEqual(actual_result, expected_result)
    expected_type = computation_types.FunctionType(None, type_spec)
    self.assertEqual(actual_type, expected_type)

  @parameterized.named_parameters(
      (
          'federated_type',
          1,
          computation_types.FederatedType(np.int64, placements.SERVER),
      ),
      ('bad_type', 1, computation_types.TensorType(np.str_)),
      (
          'tuple_larger_than_struct_type',
          (1, 2, 3.0),
          computation_types.StructType([np.int64, np.int64]),
      ),
      (
          'tuple_smaller_than_struct_type',
          (1, 2),
          computation_types.StructType([np.int64, np.int64, np.float32]),
      ),
      (
          'dict_and_struct_type_unnamed',
          {'a': 1, 'b': 2, 'c': 3.0},
          computation_types.StructType([np.int64, np.int64, np.float32]),
      ),
  )
  def test_raises_error(self, value, type_spec):
    factory = xla_computation_factory.XlaComputationFactory()

    # DO_NOT_SUBMIT: Clean up errors.
    with self.assertRaises((TypeError, ValueError, xla_client.XlaRuntimeError)):
      factory.create_constant(value, type_spec)


class CreateEmptyTupleTest(parameterized.TestCase):

  def test_returns_result(self):
    factory = xla_computation_factory.XlaComputationFactory()

    proto, actual_type = factory.create_empty_tuple()

    actual_result = _run_jax(proto, actual_type)
    expected_result = structure.Struct([])
    self.assertEqual(actual_result, expected_result)
    expected_type = computation_types.FunctionType(
        None, computation_types.StructType([])
    )
    self.assertEqual(actual_type, expected_type)


class CreateRandomUniformTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', 1, 10, computation_types.TensorType(np.int64)),
      ('float', 1.0, 10.0, computation_types.TensorType(np.float64)),
  )
  def test_returns_result(self, low, high, type_spec):
    factory = xla_computation_factory.XlaComputationFactory()

    proto, actual_type = factory.create_random_uniform(low, high, type_spec)

    actual_result = _run_jax(proto, actual_type)
    self.assertBetween(actual_result, low, high)
    expected_type = computation_types.FunctionType(None, type_spec)
    self.assertEqual(actual_type, expected_type)

  @parameterized.named_parameters(
      (
          'federated_type',
          computation_types.FederatedType(np.int32, placements.SERVER),
      ),
  )
  def test_raises_value_error(self, type_spec):
    factory = xla_computation_factory.XlaComputationFactory()

    with self.assertRaises(ValueError):
      factory.create_identity(type_spec)


class CreateIdentityTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', computation_types.TensorType(np.int64), 1),
      ('float', computation_types.TensorType(np.float64), 1.0),
      # DO_NOT_SUBMIT: Should this handle a sequence?
      # DO_NOT_SUBMIT: Should these be Python values?.
      (
          'struct_unnamed',
          computation_types.StructType([np.int64, np.int64, np.float64]),
          structure.Struct([(None, 1), (None, 2), (None, 3.0)]),
      ),
      (
          'struct_named',
          computation_types.StructType([
              ('a', np.int64),
              ('b', np.int64),
              ('c', np.float64),
          ]),
          structure.Struct([('a', 1), ('b', 2), ('c', 3.0)]),
      ),
      (
          'struct_nested',
          computation_types.StructType([
              np.int64,
              computation_types.StructType([np.int64, np.float64]),
          ]),
          structure.Struct([
              (None, 1),
              (None, structure.Struct([(None, 2), (None, 3.0)])),
          ]),
      ),
  )
  def test_returns_result(self, type_spec, arg):
    factory = xla_computation_factory.XlaComputationFactory()

    proto, actual_type = factory.create_identity(type_spec)

    actual_result = _run_jax(proto, actual_type, arg)
    expected_result = arg
    self.assertEqual(actual_result, expected_result)
    expected_type = computation_types.FunctionType(type_spec, type_spec)
    self.assertEqual(actual_type, expected_type)

  @parameterized.named_parameters(
      (
          'federated_type',
          computation_types.FederatedType(np.int32, placements.SERVER),
      ),
  )
  def test_raises_value_error(self, type_spec):
    factory = xla_computation_factory.XlaComputationFactory()

    with self.assertRaises(ValueError):
      factory.create_identity(type_spec)


class CreateAddTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', computation_types.TensorType(np.int64), 2, 3, 5),
      ('float', computation_types.TensorType(np.float64), 2.0, 3.0, 5.0),
      (
          'struct_unnamed',
          computation_types.StructType([np.int64, np.int64, np.float64]),
          structure.Struct([(None, 1), (None, 2), (None, 3.0)]),
          structure.Struct([(None, 2), (None, 2), (None, 2.0)]),
          structure.Struct([(None, 3), (None, 4), (None, 5.0)]),
      ),
      (
          'struct_named',
          computation_types.StructType([
              ('a', np.int64),
              ('b', np.int64),
              ('c', np.float64),
          ]),
          structure.Struct([('a', 1), ('b', 2), ('c', 3.0)]),
          structure.Struct([('a', 2), ('b', 2), ('c', 2.0)]),
          structure.Struct([('a', 3), ('b', 4), ('c', 5.0)]),
      ),
      (
          'struct_nested',
          computation_types.StructType([
              np.int64,
              computation_types.StructType([np.int64, np.float64]),
          ]),
          structure.Struct([
              (None, 1),
              (None, structure.Struct([(None, 2), (None, 3.0)])),
          ]),
          structure.Struct([
              (None, 2),
              (None, structure.Struct([(None, 2), (None, 2.0)])),
          ]),
          structure.Struct([
              (None, 3),
              (None, structure.Struct([(None, 4), (None, 5.0)])),
          ]),
      ),
  )
  def test_returns_result(self, type_spec, arg1, arg2, expected_result):
    factory = xla_computation_factory.XlaComputationFactory()

    proto, actual_type = factory.create_add(type_spec)

    arg = structure.Struct([(None, arg1), (None, arg2)])
    actual_result = _run_jax(proto, actual_type, arg)
    self.assertEqual(actual_result, expected_result)
    expected_type = computation_types.FunctionType(
        computation_types.StructType([type_spec, type_spec]), type_spec
    )
    self.assertEqual(actual_type, expected_type)

  @parameterized.named_parameters(
      (
          'federated_type',
          computation_types.FederatedType(np.int32, placements.SERVER),
      ),
  )
  def test_raises_value_error(self, type_spec):
    factory = xla_computation_factory.XlaComputationFactory()

    with self.assertRaises(ValueError):
      factory.create_add(type_spec)


class CreateMultiplyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', computation_types.TensorType(np.int64), 2, 3, 6),
      ('float', computation_types.TensorType(np.float64), 2.0, 3.0, 6.0),
      (
          'struct_unnamed',
          computation_types.StructType([np.int64, np.int64, np.float64]),
          structure.Struct([(None, 1), (None, 2), (None, 3.0)]),
          structure.Struct([(None, 2), (None, 2), (None, 2.0)]),
          structure.Struct([(None, 2), (None, 4), (None, 6.0)]),
      ),
      (
          'struct_named',
          computation_types.StructType([
              ('a', np.int64),
              ('b', np.int64),
              ('c', np.float64),
          ]),
          structure.Struct([('a', 1), ('b', 2), ('c', 3.0)]),
          structure.Struct([('a', 2), ('b', 2), ('c', 2.0)]),
          structure.Struct([('a', 2), ('b', 4), ('c', 6.0)]),
      ),
      (
          'struct_nested',
          computation_types.StructType([
              np.int64,
              computation_types.StructType([np.int64, np.float64]),
          ]),
          structure.Struct([
              (None, 1),
              (None, structure.Struct([(None, 2), (None, 3.0)])),
          ]),
          structure.Struct([
              (None, 2),
              (None, structure.Struct([(None, 2), (None, 2.0)])),
          ]),
          structure.Struct([
              (None, 2),
              (None, structure.Struct([(None, 4), (None, 6.0)])),
          ]),
      ),
  )
  def test_returns_result(self, type_spec, arg1, arg2, expected_result):
    factory = xla_computation_factory.XlaComputationFactory()

    proto, actual_type = factory.create_multiply(type_spec)

    arg = structure.Struct([(None, arg1), (None, arg2)])
    actual_result = _run_jax(proto, actual_type, arg)
    self.assertEqual(actual_result, expected_result)
    expected_type = computation_types.FunctionType(
        computation_types.StructType([type_spec, type_spec]), type_spec
    )
    self.assertEqual(actual_type, expected_type)

  @parameterized.named_parameters(
      (
          'federated_type',
          computation_types.FederatedType(np.int32, placements.SERVER),
      ),
  )
  def test_raises_value_error(self, type_spec):
    factory = xla_computation_factory.XlaComputationFactory()

    with self.assertRaises(ValueError):
      factory.create_multiply(type_spec)


class CreateMinTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', computation_types.TensorType(np.int64), 2, 3, 2),
      ('float', computation_types.TensorType(np.float64), 2.0, 3.0, 2.0),
      (
          'struct_unnamed',
          computation_types.StructType([np.int64, np.int64, np.float64]),
          structure.Struct([(None, 1), (None, 2), (None, 3.0)]),
          structure.Struct([(None, 2), (None, 2), (None, 2.0)]),
          structure.Struct([(None, 1), (None, 2), (None, 2.0)]),
      ),
      (
          'struct_named',
          computation_types.StructType([
              ('a', np.int64),
              ('b', np.int64),
              ('c', np.float64),
          ]),
          structure.Struct([('a', 1), ('b', 2), ('c', 3.0)]),
          structure.Struct([('a', 2), ('b', 2), ('c', 2.0)]),
          structure.Struct([('a', 1), ('b', 2), ('c', 2.0)]),
      ),
      (
          'struct_nested',
          computation_types.StructType([
              np.int64,
              computation_types.StructType([np.int64, np.float64]),
          ]),
          structure.Struct([
              (None, 1),
              (None, structure.Struct([(None, 2), (None, 3.0)])),
          ]),
          structure.Struct([
              (None, 2),
              (None, structure.Struct([(None, 2), (None, 2.0)])),
          ]),
          structure.Struct([
              (None, 1),
              (None, structure.Struct([(None, 2), (None, 2.0)])),
          ]),
      ),
  )
  def test_returns_result(self, type_spec, arg1, arg2, expected_result):
    factory = xla_computation_factory.XlaComputationFactory()

    proto, actual_type = factory.create_min(type_spec)

    arg = structure.Struct([(None, arg1), (None, arg2)])
    actual_result = _run_jax(proto, actual_type, arg)
    self.assertEqual(actual_result, expected_result)
    expected_type = computation_types.FunctionType(
        computation_types.StructType([type_spec, type_spec]), type_spec
    )
    self.assertEqual(actual_type, expected_type)

  @parameterized.named_parameters(
      (
          'federated_type',
          computation_types.FederatedType(np.int32, placements.SERVER),
      ),
  )
  def test_raises_value_error(self, type_spec):
    factory = xla_computation_factory.XlaComputationFactory()

    with self.assertRaises(ValueError):
      factory.create_min(type_spec)


class CreateMaxTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', computation_types.TensorType(np.int64), 2, 3, 3),
      ('float', computation_types.TensorType(np.float64), 2.0, 3.0, 3.0),
      (
          'struct_unnamed',
          computation_types.StructType([np.int64, np.int64, np.float64]),
          structure.Struct([(None, 1), (None, 2), (None, 3.0)]),
          structure.Struct([(None, 2), (None, 2), (None, 2.0)]),
          structure.Struct([(None, 2), (None, 2), (None, 3.0)]),
      ),
      (
          'struct_named',
          computation_types.StructType([
              ('a', np.int64),
              ('b', np.int64),
              ('c', np.float64),
          ]),
          structure.Struct([('a', 1), ('b', 2), ('c', 3.0)]),
          structure.Struct([('a', 2), ('b', 2), ('c', 2.0)]),
          structure.Struct([('a', 2), ('b', 2), ('c', 3.0)]),
      ),
      (
          'struct_nested',
          computation_types.StructType([
              np.int64,
              computation_types.StructType([np.int64, np.float64]),
          ]),
          structure.Struct([
              (None, 1),
              (None, structure.Struct([(None, 2), (None, 3.0)])),
          ]),
          structure.Struct([
              (None, 2),
              (None, structure.Struct([(None, 2), (None, 2.0)])),
          ]),
          structure.Struct([
              (None, 2),
              (None, structure.Struct([(None, 2), (None, 3.0)])),
          ]),
      ),
  )
  def test_returns_result(self, type_spec, arg1, arg2, expected_result):
    factory = xla_computation_factory.XlaComputationFactory()

    proto, actual_type = factory.create_max(type_spec)

    arg = structure.Struct([(None, arg1), (None, arg2)])
    actual_result = _run_jax(proto, actual_type, arg)
    self.assertEqual(actual_result, expected_result)
    expected_type = computation_types.FunctionType(
        computation_types.StructType([type_spec, type_spec]), type_spec
    )
    self.assertEqual(actual_type, expected_type)

  @parameterized.named_parameters(
      (
          'federated_type',
          computation_types.FederatedType(np.int32, placements.SERVER),
      ),
  )
  def test_raises_value_error(self, type_spec):
    factory = xla_computation_factory.XlaComputationFactory()

    with self.assertRaises(ValueError):
      factory.create_max(type_spec)


if __name__ == '__main__':
  absltest.main()
