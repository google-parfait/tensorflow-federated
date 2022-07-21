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

import collections

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation_utils
from tensorflow_federated.python.core.impl.types import computation_types

TestNamedTuple = collections.namedtuple('TestTuple', ['x', 'y'])


def _federated_computation_serializer(fn, parameter_name, parameter_type):
  unpack_arguments = function_utils.create_argument_unpacking_fn(
      fn, parameter_type)
  fn_gen = federated_computation_utils.federated_computation_serializer(
      parameter_name, parameter_type, context_stack_impl.context_stack)
  args, kwargs = unpack_arguments(next(fn_gen))
  result = fn(*args, **kwargs)
  return fn_gen.send(result)


class FnToBuildingBlockTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('nested_fn_same',
       lambda f, x: f(f(x)),
       computation_types.StructType([
           ('f', computation_types.FunctionType(tf.int32, tf.int32)),
           ('x', tf.int32)]),
       '(FEDERATED_foo -> (let fc_FEDERATED_symbol_0=FEDERATED_foo.f(FEDERATED_foo.x),fc_FEDERATED_symbol_1=FEDERATED_foo.f(fc_FEDERATED_symbol_0) in fc_FEDERATED_symbol_1))'),
      ('nested_fn_different',
       lambda f, g, x: f(g(x)),
       computation_types.StructType([
           ('f', computation_types.FunctionType(tf.int32, tf.int32)),
           ('g', computation_types.FunctionType(tf.int32, tf.int32)),
           ('x', tf.int32)]),
       '(FEDERATED_foo -> (let fc_FEDERATED_symbol_0=FEDERATED_foo.g(FEDERATED_foo.x),fc_FEDERATED_symbol_1=FEDERATED_foo.f(fc_FEDERATED_symbol_0) in fc_FEDERATED_symbol_1))'),
      ('selection',
       lambda x: (x[1], x[0]),
       computation_types.StructType([tf.int32, tf.int32]),
       '(FEDERATED_foo -> <FEDERATED_foo[1],FEDERATED_foo[0]>)'),
      ('constant', lambda: 'stuff', None, '( -> (let fc_FEDERATED_symbol_0=comp#'))
  # pyformat: enable
  def test_returns_result(self, fn, parameter_type, fn_str):
    parameter_name = 'foo' if parameter_type is not None else None
    result, _ = _federated_computation_serializer(fn, parameter_name,
                                                  parameter_type)
    self.assertStartsWith(str(result), fn_str)

  # pyformat: disable
  @parameterized.named_parameters(
      ('tuple',
       lambda x: (x[1], x[0]),
       computation_types.StructType([tf.int32, tf.float32]),
       computation_types.StructWithPythonType([
           (None, tf.float32), (None, tf.int32)], tuple)),
      ('list',
       lambda x: [x[1], x[0]],
       computation_types.StructType([tf.int32, tf.float32]),
       computation_types.StructWithPythonType([
           (None, tf.float32), (None, tf.int32)], list)),
      ('odict',
       lambda x: collections.OrderedDict([('A', x[1]), ('B', x[0])]),
       computation_types.StructType([tf.int32, tf.float32]),
       computation_types.StructWithPythonType([
           ('A', tf.float32), ('B', tf.int32)], collections.OrderedDict)),
      ('namedtuple',
       lambda x: TestNamedTuple(x=x[1], y=x[0]),
       computation_types.StructType([tf.int32, tf.float32]),
       computation_types.StructWithPythonType([
           ('x', tf.float32), ('y', tf.int32)], TestNamedTuple)),
  )
  # pyformat: enable
  def test_returns_result_with_py_container(self, fn, parameter_type,
                                            expected_result_type):
    _, type_signature = _federated_computation_serializer(
        fn, 'foo', parameter_type)
    self.assertIs(type(type_signature.result), type(expected_result_type))
    self.assertIs(type_signature.result.python_container,
                  expected_result_type.python_container)
    self.assertEqual(type_signature.result, expected_result_type)


if __name__ == '__main__':
  absltest.main()
