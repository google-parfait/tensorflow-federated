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

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import federated_computation_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.utils import function_utils


TestNamedTuple = collections.namedtuple('TestTuple', ['x', 'y'])


class ZeroOrOneArgFnToBuildingBlockTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('nested_fn_same',
       lambda f, x: f(f(x)),
       computation_types.NamedTupleType([
           ('f', computation_types.FunctionType(tf.int32, tf.int32)),
           ('x', tf.int32)]),
       '(FEDERATED_foo -> (let fc_FEDERATED_symbol_0=FEDERATED_foo.f(FEDERATED_foo.x),fc_FEDERATED_symbol_1=FEDERATED_foo.f(fc_FEDERATED_symbol_0) in fc_FEDERATED_symbol_1))'),
      ('nested_fn_different',
       lambda f, g, x: f(g(x)),
       computation_types.NamedTupleType([
           ('f', computation_types.FunctionType(tf.int32, tf.int32)),
           ('g', computation_types.FunctionType(tf.int32, tf.int32)),
           ('x', tf.int32)]),
       '(FEDERATED_foo -> (let fc_FEDERATED_symbol_0=FEDERATED_foo.g(FEDERATED_foo.x),fc_FEDERATED_symbol_1=FEDERATED_foo.f(fc_FEDERATED_symbol_0) in fc_FEDERATED_symbol_1))'),
      ('selection',
       lambda x: (x[1], x[0]),
       computation_types.NamedTupleType([tf.int32, tf.int32]),
       '(FEDERATED_foo -> <FEDERATED_foo[1],FEDERATED_foo[0]>)'),
      ('constant', lambda: 'stuff', None, '( -> comp#'))
  # pyformat: enable
  def test_zero_or_one_arg_fn_to_building_block(self, fn, parameter_type,
                                                fn_str):
    parameter_name = 'foo' if parameter_type is not None else None
    fn = function_utils.wrap_as_zero_or_one_arg_callable(fn, parameter_type)
    result, _ = federated_computation_utils.zero_or_one_arg_fn_to_building_block(
        fn, parameter_name, parameter_type, context_stack_impl.context_stack)
    self.assertStartsWith(str(result), fn_str)

  # pyformat: disable
  @parameterized.named_parameters(
      ('tuple_result',
       lambda x: (x[1], x[0]),
       computation_types.NamedTupleType([tf.int32, tf.float32]),
       computation_types.NamedTupleTypeWithPyContainerType([
           (None, tf.float32), (None, tf.int32)], tuple)),
      ('list_result',
       lambda x: [x[1], x[0]],
       computation_types.NamedTupleType([tf.int32, tf.float32]),
       computation_types.NamedTupleTypeWithPyContainerType([
           (None, tf.float32), (None, tf.int32)], list)),
      ('odict_result',
       lambda x: collections.OrderedDict([('A', x[1]), ('B', x[0])]),
       computation_types.NamedTupleType([tf.int32, tf.float32]),
       computation_types.NamedTupleTypeWithPyContainerType([
           ('A', tf.float32), ('B', tf.int32)], collections.OrderedDict)),
      ('namedtuple_result',
       lambda x: TestNamedTuple(x=x[1], y=x[0]),
       computation_types.NamedTupleType([tf.int32, tf.float32]),
       computation_types.NamedTupleTypeWithPyContainerType([
           ('x', tf.float32), ('y', tf.int32)], TestNamedTuple)),
  )
  # pyformat: enable
  def test_py_container_args(self, fn, parameter_type, exepcted_result_type):
    parameter_name = 'foo'
    fn = function_utils.wrap_as_zero_or_one_arg_callable(fn, parameter_type)
    _, type_signature = federated_computation_utils.zero_or_one_arg_fn_to_building_block(
        fn, parameter_name, parameter_type, context_stack_impl.context_stack)
    self.assertIs(type(type_signature.result), type(exepcted_result_type))
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            type_signature.result),
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            exepcted_result_type))
    self.assertEqual(type_signature.result, exepcted_result_type)


if __name__ == '__main__':
  absltest.main()
