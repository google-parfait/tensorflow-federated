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

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_transformations


def _convert_tensor_to_float(type_spec):
  if type_spec.is_tensor():
    return computation_types.TensorType(tf.float32, shape=type_spec.shape), True
  return type_spec, False


def _convert_abstract_type_to_tensor(type_spec):
  if type_spec.is_abstract():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_placement_type_to_tensor(type_spec):
  if type_spec.is_placement():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_function_to_tensor(type_spec):
  if type_spec.is_function():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_federated_to_tensor(type_spec):
  if type_spec.is_federated():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_sequence_to_tensor(type_spec):
  if type_spec.is_sequence():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


def _convert_tuple_to_tensor(type_spec):
  if type_spec.is_struct():
    return computation_types.TensorType(tf.float32), True
  return type_spec, False


class TransformTypePostorderTest(absltest.TestCase):

  def test_raises_on_none_type(self):
    with self.assertRaises(TypeError):
      type_transformations.transform_type_postorder(None, lambda x: x)

  def test_raises_on_none_function(self):
    with self.assertRaises(TypeError):
      type_transformations.transform_type_postorder(
          computation_types.TensorType(tf.int32), None)

  def test_raises_on_non_type_first_arg(self):
    with self.assertRaises(TypeError):
      type_transformations.transform_type_postorder(tf.int32, None)

  def test_transforms_tensor(self):
    orig_type = computation_types.TensorType(tf.int32)
    expected_type = computation_types.TensorType(tf.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_transforms_federated_type(self):
    orig_type = computation_types.FederatedType(tf.int32,
                                                placement_literals.CLIENTS)
    expected_type = computation_types.FederatedType(tf.float32,
                                                    placement_literals.CLIENTS)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_recurses_under_federated_type(self):
    orig_type = computation_types.FederatedType([tf.int32],
                                                placement_literals.CLIENTS)
    expected_type = computation_types.FederatedType([tf.float32],
                                                    placement_literals.CLIENTS)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_updates_mutated_bit_at_federated(self):
    orig_type = computation_types.FederatedType(tf.int32,
                                                placement_literals.CLIENTS)
    _, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_federated_to_tensor)
    self.assertTrue(mutated)

  def test_transforms_sequence(self):
    orig_type = computation_types.SequenceType(tf.int32)
    expected_type = computation_types.SequenceType(tf.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_recurses_under_sequence(self):
    orig_type = computation_types.SequenceType([tf.int32])
    expected_type = computation_types.SequenceType([tf.float32])
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_updates_mutated_bit_at_sequence(self):
    orig_type = computation_types.SequenceType(tf.int32)
    _, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_sequence_to_tensor)
    self.assertTrue(mutated)

  def test_transforms_function(self):
    orig_type = computation_types.FunctionType(tf.int32, tf.int64)
    expected_type = computation_types.FunctionType(tf.float32, tf.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_recurses_under_function(self):
    orig_type = computation_types.FunctionType([tf.int32], tf.int64)
    expected_type = computation_types.FunctionType([tf.float32], tf.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_updates_mutated_bit_at_function(self):
    orig_type = computation_types.FunctionType(tf.int32, tf.int64)
    _, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_function_to_tensor)
    self.assertTrue(mutated)

  def test_transforms_unnamed_tuple_type_preserving_tuple_container(self):
    orig_type = computation_types.NamedTupleTypeWithPyContainerType(
        [tf.int32, tf.float64], tuple)
    expected_type = computation_types.NamedTupleTypeWithPyContainerType(
        [tf.float32, tf.float32], tuple)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_transforms_unnamed_tuple_type(self):
    orig_type = computation_types.NamedTupleType([tf.int32, tf.float64])
    expected_type = computation_types.NamedTupleType([tf.float32, tf.float32])
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_updates_mutated_bit_at_tuple(self):
    orig_type = computation_types.NamedTupleType([tf.int32, tf.float64])
    _, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tuple_to_tensor)
    self.assertTrue(mutated)

  def test_transforms_named_tuple_type(self):
    orig_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  ('b', tf.float64)])
    expected_type = computation_types.NamedTupleType([('a', tf.float32),
                                                      ('b', tf.float32)])
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_recurses_under_named_tuple_type(self):
    orig_type = computation_types.NamedTupleType([[('a', tf.int32),
                                                   ('b', tf.float64)]])
    expected_type = computation_types.NamedTupleType([[('a', tf.float32),
                                                       ('b', tf.float32)]])
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_transforms_named_tuple_type_preserving_tuple_container(self):
    orig_type = computation_types.NamedTupleTypeWithPyContainerType(
        [('a', tf.int32), ('b', tf.float64)], dict)
    expected_type = computation_types.NamedTupleTypeWithPyContainerType(
        [('a', tf.float32), ('b', tf.float32)], dict)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_tensor_to_float)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_transforms_abstract_type(self):
    orig_type = computation_types.AbstractType('T')
    expected_type = computation_types.TensorType(tf.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_placement_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)

  def test_transforms_placement_type(self):
    orig_type = computation_types.PlacementType()
    expected_type = computation_types.TensorType(tf.float32)
    result_type, mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_placement_type_to_tensor)
    noop_type, not_mutated = type_transformations.transform_type_postorder(
        orig_type, _convert_abstract_type_to_tensor)
    self.assertEqual(result_type, expected_type)
    self.assertEqual(noop_type, orig_type)
    self.assertTrue(mutated)
    self.assertFalse(not_mutated)


class VisitPreorderTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters([
      ('abstract_type', computation_types.AbstractType('T'), 1),
      ('nested_federated_type',
       computation_types.FederatedType(
           computation_types.FederatedType(
               computation_types.FederatedType(
                   tf.int32, placement_literals.CLIENTS),
               placement_literals.CLIENTS),
           placement_literals.CLIENTS),
       4),
      ('nested_function_type',
       computation_types.FunctionType(
           computation_types.FunctionType(
               computation_types.FunctionType(tf.int32, tf.int32),
               tf.int32),
           tf.int32),
       7),
      ('nested_sequence_type',
       computation_types.SequenceType(
           computation_types.SequenceType(
               computation_types.SequenceType(tf.int32))),
       4),
      ('named_tuple_type',
       computation_types.NamedTupleType([
           tf.int32,
           tf.bool,
           computation_types.SequenceType(tf.int32)]),
       5),
      ('placement_type', computation_types.PlacementType(), 1),
  ])
  # pyformat: enable
  def test_preorder_call_count(self, type_signature, expected_count):

    class Counter(object):
      k = 0

    def _count_hits(given_type, arg):
      del given_type  # Unused.
      Counter.k += 1
      return arg

    type_transformations.visit_preorder(type_signature, _count_hits, None)
    actual_count = Counter.k
    self.assertEqual(actual_count, expected_count)


if __name__ == '__main__':
  absltest.main()
