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

import collections
from absl.testing import absltest
from jax.lib.xla_bridge import xla_client
import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.backends.xla import compiler
from tensorflow_federated.python.core.backends.xla import runtime


class CompilerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._factory = compiler.XlaComputationFactory()

  def test_create_constant_from_scalar_float32(self):
    constant_type = computation_types.TensorType(np.float32)
    comp, comp_type = self._factory.create_constant_from_scalar(
        10.0, constant_type)
    result = self._run_comp(comp, comp_type)
    self.assertEqual(result, 10.0)

  def test_create_constant_from_scalar_int32x3(self):
    constant_type = computation_types.TensorType(np.int32, [3])
    comp, comp_type = self._factory.create_constant_from_scalar(
        10, constant_type)
    result = self._run_comp(comp, comp_type)
    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.dtype, np.int32)
    self.assertListEqual(list(result.shape), [3])
    self.assertEqual(list(result.flatten()), [10, 10, 10])

  def test_create_constant_from_scalar_int32x3x2(self):
    constant_type = computation_types.TensorType(np.int32, [3, 2])
    comp, comp_type = self._factory.create_constant_from_scalar(
        10, constant_type)
    result = self._run_comp(comp, comp_type)
    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.dtype, np.int32)
    self.assertListEqual(list(result.shape), [3, 2])
    self.assertEqual(list(result.flatten()), [10, 10, 10, 10, 10, 10])

  def test_create_constant_from_scalar_int32_struct(self):
    constant_type = computation_types.to_type(
        collections.OrderedDict([('a', np.int32), ('b', np.int32)]))
    comp, comp_type = self._factory.create_constant_from_scalar(
        10, constant_type)
    result = self._run_comp(comp, comp_type)
    self.assertEqual(str(result), '<a=10,b=10>')

  def test_create_constant_from_scalar_float32_nested_struct(self):
    constant_type = computation_types.to_type(
        collections.OrderedDict([
            ('a', np.float32),
            ('b', collections.OrderedDict([('c', np.float32)]))
        ]))
    comp, comp_type = self._factory.create_constant_from_scalar(
        10, constant_type)
    result = self._run_comp(comp, comp_type)
    self.assertEqual(str(result), '<a=10.0,b=<c=10.0>>')

  def test_create_plus_operator_int32(self):
    arg_type = computation_types.to_type(np.int32)
    comp, comp_type = self._factory.create_plus_operator(arg_type)
    arg = structure.Struct([(None, np.int32(2)), (None, np.int32(3))])
    result = self._run_comp(comp, comp_type, arg=arg)
    self.assertEqual(result, 5)

  def test_create_plus_operator_nested_structure(self):
    arg_type = computation_types.to_type(
        collections.OrderedDict([('a', np.int32), ('b', np.float32),
                                 ('c',
                                  collections.OrderedDict([('d', np.float32),
                                                           ('e', np.int32)]))]))
    comp, comp_type = self._factory.create_plus_operator(arg_type)
    arg1 = collections.OrderedDict([
        ('a', np.int32(2)), ('b', np.float32(3.0)),
        ('c',
         collections.OrderedDict([('d', np.float32(4.0)), ('e', np.int32(5))]))
    ])
    arg2 = collections.OrderedDict([
        ('a', np.int32(5)), ('b', np.float32(6.0)),
        ('c',
         collections.OrderedDict([('d', np.float32(7.0)), ('e', np.int32(8))]))
    ])
    arg = structure.Struct([(None, arg1), (None, arg2)])
    result = self._run_comp(comp, comp_type, arg=arg)
    self.assertEqual(str(result), '<a=7,b=9.0,c=<d=11.0,e=13>>')

  def test_create_multiply_operator_int32(self):
    arg_type = computation_types.to_type(np.int32)
    comp, comp_type = self._factory.create_multiply_operator(arg_type)
    arg = structure.Struct([(None, np.int32(2)), (None, np.int32(3))])
    result = self._run_comp(comp, comp_type, arg=arg)
    self.assertEqual(result, 6)

  def test_create_scalar_multiply_operator_float32(self):
    operand_type = computation_types.to_type(np.float32)
    scalar_type = computation_types.to_type(np.float32)
    comp, comp_type = self._factory.create_scalar_multiply_operator(
        operand_type, scalar_type)
    operand = np.float32(10.0)
    scalar = np.float32(5.0)
    arg = structure.Struct([(None, operand), (None, scalar)])
    result = self._run_comp(comp, comp_type, arg=arg)
    self.assertEqual(result, 50.0)

  def test_create_scalar_multiply_operator_2xfloat32(self):
    operand_type = computation_types.to_type(
        collections.OrderedDict([('a', np.float32), ('b', np.float32)]))
    scalar_type = computation_types.to_type(np.float32)
    comp, comp_type = self._factory.create_scalar_multiply_operator(
        operand_type, scalar_type)
    operand = collections.OrderedDict([('a', np.float32(10.0)),
                                       ('b', np.float32(11.0))])
    scalar = np.float32(5.0)
    arg = structure.Struct([(None, operand), (None, scalar)])
    result = self._run_comp(comp, comp_type, arg=arg)
    self.assertEqual(str(result), '<a=50.0,b=55.0>')

  def _run_comp(self, comp_pb, comp_type, arg=None):
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertIsInstance(comp_type, computation_types.FunctionType)
    backend = xla_client.get_local_backend(None)
    comp_callable = runtime.ComputationCallable(comp_pb, comp_type, backend)
    arg_list = []
    if arg is not None:
      arg_list.append(arg)
    return comp_callable(*arg_list)


if __name__ == '__main__':
  absltest.main()
