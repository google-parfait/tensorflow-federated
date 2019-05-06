# Lint as: python3
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
"""Tests for value_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import federated_computation_utils
from tensorflow_federated.python.core.impl import function_utils


# Aliases for convenience in unittests.
NamedTupleTypeWithPyContainerType = computation_types.NamedTupleTypeWithPyContainerType
TensorType = computation_types.TensorType
TestNamedTuple = collections.namedtuple('TestTuple', ['x', 'y'])

SCALAR_INT_TYPE = TensorType(tf.int32, [])
SCALAR_FLOAT_TYPE = TensorType(tf.float32, [])


class ZeroOrOneArgFnToBuildingBlockTest(parameterized.TestCase):

  @parameterized.parameters(
      (lambda f, x: f(f(x)), [
          ('f', computation_types.FunctionType(tf.int32, tf.int32)),
          ('x', tf.int32)
      ],
       '(FEDERATED_foo -> FEDERATED_foo.f(FEDERATED_foo.f(FEDERATED_foo.x)))'),
      (lambda f, g, x: f(g(x)), [
          ('f', computation_types.FunctionType(tf.int32, tf.int32)),
          ('g', computation_types.FunctionType(tf.int32, tf.int32)),
          ('x', tf.int32)
      ],
       '(FEDERATED_foo -> FEDERATED_foo.f(FEDERATED_foo.g(FEDERATED_foo.x)))'),
      (lambda x: (x[1], x[0]), (tf.int32, tf.int32),
       '(FEDERATED_foo -> <FEDERATED_foo[1],FEDERATED_foo[0]>)'),
      (lambda: 'stuff', None, 'comp#'))
  def test_multiple_args(self, fn, parameter_type, fn_regex):
    parameter_name = 'foo'
    parameter_type = computation_types.to_type(parameter_type)
    fn = function_utils.wrap_as_zero_or_one_arg_callable(fn, parameter_type)
    result, _ = federated_computation_utils.zero_or_one_arg_fn_to_building_block(
        fn, parameter_name, parameter_type, context_stack_impl.context_stack)
    self.assertStartsWith(str(result), fn_regex)

  @parameterized.named_parameters(
      ('tuple_result', lambda x: (x[1], x[0]), (tf.int32, tf.float32),
       NamedTupleTypeWithPyContainerType([(None, SCALAR_FLOAT_TYPE),
                                          (None, SCALAR_INT_TYPE)], tuple)),
      ('list_result', lambda x: [x[1], x[0]], (tf.int32, tf.float32),
       NamedTupleTypeWithPyContainerType([(None, SCALAR_FLOAT_TYPE),
                                          (None, SCALAR_INT_TYPE)], list)),
      ('odict_result',
       lambda x: collections.OrderedDict([('A', x[1]), ('B', x[0])]),
       (tf.int32, tf.float32),
       NamedTupleTypeWithPyContainerType([('A', SCALAR_FLOAT_TYPE),
                                          ('B', SCALAR_INT_TYPE)],
                                         collections.OrderedDict)),
      ('namedtuple_result', lambda x: TestNamedTuple(x=x[1], y=x[0]),
       (tf.int32, tf.float32),
       NamedTupleTypeWithPyContainerType([('x', SCALAR_FLOAT_TYPE),
                                          ('y', SCALAR_INT_TYPE)],
                                         TestNamedTuple)),
  )
  def test_py_container_args(self, fn, parameter_type, result_type):
    parameter_name = 'foo'
    parameter_type = computation_types.to_type(parameter_type)
    fn = function_utils.wrap_as_zero_or_one_arg_callable(fn, parameter_type)
    _, annotated_type = federated_computation_utils.zero_or_one_arg_fn_to_building_block(
        fn, parameter_name, parameter_type, context_stack_impl.context_stack)
    self.assertIs(type(annotated_type.result), type(result_type))
    self.assertIs(
        NamedTupleTypeWithPyContainerType.get_container_type(
            annotated_type.result),
        NamedTupleTypeWithPyContainerType.get_container_type(result_type))
    self.assertEqual(annotated_type.result, result_type)


if __name__ == '__main__':
  absltest.main()
