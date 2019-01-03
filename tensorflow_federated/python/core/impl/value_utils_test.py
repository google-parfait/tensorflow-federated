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
"""Implementations of the abstract interface Value in api/value_base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import absltest
from absl.testing import parameterized
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import value_impl
from tensorflow_federated.python.core.impl import value_utils


_context_stack = context_stack_impl.context_stack


class ValueUtilsTest(parameterized.TestCase):

  def test_two_tuple_zip_with_client_non_all_equal_int_and_bool(self):
    test_ref = computation_building_blocks.Reference(
        'test',
        computation_types.NamedTupleType([
            computation_types.FederatedType(tf.int32, placements.CLIENTS),
            computation_types.FederatedType(tf.bool, placements.CLIENTS, True)
        ]))
    zipped = value_utils.zip_two_tuple(
        value_impl.to_value(test_ref, None, _context_stack), _context_stack)
    self.assertEqual(str(zipped.type_signature), '{<int32,bool>}@CLIENTS')

  def test_two_tuple_zip_with_client_all_equal_int_and_bool(self):
    test_ref = computation_building_blocks.Reference(
        'test',
        computation_types.NamedTupleType([
            computation_types.FederatedType(tf.int32, placements.CLIENTS, True),
            computation_types.FederatedType(tf.bool, placements.CLIENTS, True)
        ]))
    zipped = value_utils.zip_two_tuple(
        value_impl.to_value(test_ref, None, _context_stack), _context_stack)
    self.assertEqual(str(zipped.type_signature), '<int32,bool>@CLIENTS')

  def test_two_tuple_zip_fails_bad_args(self):
    server_test_ref = computation_building_blocks.Reference(
        'test',
        computation_types.NamedTupleType([
            computation_types.FederatedType(tf.int32, placements.CLIENTS, True),
            computation_types.FederatedType(tf.bool, placements.SERVER, True)
        ]))
    with self.assertRaisesRegexp(TypeError, 'should be placed at CLIENTS'):
      _ = value_utils.zip_two_tuple(
          value_impl.to_value(server_test_ref, None, _context_stack),
          _context_stack)
    client_test_ref = computation_building_blocks.Reference(
        'test',
        computation_types.NamedTupleType([
            computation_types.FederatedType(tf.int32, placements.CLIENTS, True),
            computation_types.FederatedType(tf.bool, placements.CLIENTS, True)
        ]))
    with self.assertRaisesRegexp(TypeError, '(Expected).*(Value)'):
      _ = value_utils.zip_two_tuple(client_test_ref, _context_stack)
    three_tuple_test_ref = computation_building_blocks.Reference(
        'three_tuple_test',
        computation_types.NamedTupleType([
            computation_types.FederatedType(tf.int32, placements.CLIENTS, True),
            computation_types.FederatedType(tf.int32, placements.CLIENTS, True),
            computation_types.FederatedType(tf.int32, placements.CLIENTS, True)
        ]))
    with self.assertRaisesRegexp(ValueError, 'must be a 2-tuple'):
      _ = value_utils.zip_two_tuple(
          value_impl.to_value(three_tuple_test_ref, None, _context_stack),
          _context_stack)

  def test_flatten_func_comp_raises_typeerror(self):
    input_reference = computation_building_blocks.Reference(
        'test', [tf.int32] * 5)
    input_function = computation_building_blocks.Lambda(
        'test', input_reference.type_signature, input_reference)
    type_to_add = computation_types.NamedTupleType([tf.int32])
    with self.assertRaisesRegexp(TypeError, '(Expected).*(Value)'):
      _ = value_utils.flatten_first_index(input_function, type_to_add,
                                          _context_stack)

  @parameterized.named_parameters(
      [('test_n_' + str(x), x) for x in range(2, 10)])
  def test_flatten_function(self, n):
    input_reference = computation_building_blocks.Reference(
        'test', [tf.int32] * n)
    input_function = computation_building_blocks.Lambda(
        'test', input_reference.type_signature, input_reference)
    type_to_add = computation_types.to_type(tf.int32)
    input_type = computation_types.NamedTupleType(
        [input_reference.type_signature, type_to_add])
    desired_output_type = computation_types.to_type([tf.int32] * (n + 1))
    desired_function_type = computation_types.FunctionType(
        input_type, desired_output_type)
    new_func = value_utils.flatten_first_index(
        value_impl.to_value(input_function, None, _context_stack), type_to_add,
        _context_stack)
    self.assertEqual(str(new_func.type_signature), str(desired_function_type))


if __name__ == '__main__':
  absltest.main()
