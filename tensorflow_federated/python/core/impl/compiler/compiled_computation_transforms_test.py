# Copyright 2019, The TensorFlow Federated Authors.
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

import tensorflow as tf

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import compiled_computation_transforms
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.compiler import test_utils as compiler_test_utils
from tensorflow_federated.python.core.impl.types import computation_types


def _create_compiled_computation(py_fn, parameter_type):
  proto, type_signature = tensorflow_computation_factory.create_computation_for_py_fn(
      py_fn, parameter_type)
  return building_blocks.CompiledComputation(
      proto, type_signature=type_signature)


class TensorFlowOptimizerTest(test_case.TestCase):

  def test_should_transform_compiled_computation(self):
    tuple_type = computation_types.TensorType(tf.int32)
    compiled_computation = building_block_factory.create_compiled_identity(
        tuple_type)
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transforms.TensorFlowOptimizer(config)
    self.assertTrue(tf_optimizer.should_transform(compiled_computation))

  def test_should_not_transform_reference(self):
    reference = building_blocks.Reference('x', tf.int32)
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transforms.TensorFlowOptimizer(config)
    self.assertFalse(tf_optimizer.should_transform(reference))

  def test_transform_compiled_computation_returns_compiled_computation(self):
    tuple_type = computation_types.TensorType(tf.int32)
    compiled_computation = building_block_factory.create_compiled_identity(
        tuple_type)
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transforms.TensorFlowOptimizer(config)
    transformed_comp, mutated = tf_optimizer.transform(compiled_computation)
    self.assertTrue(mutated)
    self.assertIsInstance(transformed_comp, building_blocks.CompiledComputation)
    self.assertTrue(transformed_comp.proto.tensorflow.HasField('parameter'))
    self.assertFalse(transformed_comp.proto.tensorflow.initialize_op)

  def test_transform_compiled_computation_returns_compiled_computation_without_empty_fields(
      self):
    compiled_computation = building_block_factory.create_compiled_no_arg_empty_tuple_computation(
    )
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transforms.TensorFlowOptimizer(config)
    transformed_comp, mutated = tf_optimizer.transform(compiled_computation)
    self.assertTrue(mutated)
    self.assertIsInstance(transformed_comp, building_blocks.CompiledComputation)
    self.assertFalse(transformed_comp.proto.tensorflow.HasField('parameter'))
    self.assertFalse(transformed_comp.proto.tensorflow.initialize_op)

  def test_transform_compiled_computation_semantic_equivalence(self):
    tuple_type = computation_types.TensorType(tf.int32)
    compiled_computation = building_block_factory.create_compiled_identity(
        tuple_type)
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transforms.TensorFlowOptimizer(config)
    transformed_comp, mutated = tf_optimizer.transform(compiled_computation)
    self.assertTrue(mutated)
    self.assertIsInstance(transformed_comp, building_blocks.CompiledComputation)
    zero_before_transform = compiler_test_utils.run_tensorflow(
        compiled_computation.proto, 0)
    zero_after_transform = compiler_test_utils.run_tensorflow(
        transformed_comp.proto, 0)
    self.assertEqual(zero_before_transform, zero_after_transform)


class AddUniqueIDsTest(test_case.TestCase):

  def test_should_transform_compiled_tf_computation(self):
    tuple_type = computation_types.TensorType(tf.int32)
    compiled_computation = building_block_factory.create_compiled_identity(
        tuple_type)
    self.assertTrue(
        compiled_computation_transforms.AddUniqueIDs().should_transform(
            compiled_computation))

  def test_should_not_transform_non_compiled_computations(self):
    reference = building_blocks.Reference('x', tf.int32)
    self.assertFalse(
        compiled_computation_transforms.AddUniqueIDs().should_transform(
            reference))

  def test_transform_same_compiled_computation_different_type_signature(self):
    # First create no-arg computation that returns an empty tuple. This will
    # be compared against a omputation that returns a nested empty tuple, which
    # should produce a different ID.
    empty_tuple_computation = building_block_factory.create_tensorflow_unary_operator(
        lambda x: (), operand_type=computation_types.StructType([]))
    add_ids = compiled_computation_transforms.AddUniqueIDs()
    first_transformed_comp, mutated = add_ids.transform(empty_tuple_computation)
    self.assertTrue(mutated)
    self.assertIsInstance(first_transformed_comp,
                          building_blocks.CompiledComputation)
    self.assertTrue(
        first_transformed_comp.proto.tensorflow.HasField('cache_key'))
    self.assertNotEqual(first_transformed_comp.proto.tensorflow.cache_key.id, 0)
    # Now create the same NoOp tf.Graph, but with a different binding and
    # type_signature.
    nested_empty_tuple_computation = building_block_factory.create_tensorflow_unary_operator(
        lambda x: ((),), operand_type=computation_types.StructType([]))
    second_transformed_comp, mutated = add_ids.transform(
        nested_empty_tuple_computation)
    self.assertTrue(mutated)
    self.assertIsInstance(second_transformed_comp,
                          building_blocks.CompiledComputation)
    self.assertTrue(
        second_transformed_comp.proto.tensorflow.HasField('cache_key'))
    self.assertNotEqual(second_transformed_comp.proto.tensorflow.cache_key.id,
                        0)
    # Assert the IDs are different based on the type signture.
    self.assertNotEqual(first_transformed_comp.proto.tensorflow.cache_key.id,
                        second_transformed_comp.proto.tensorflow.cache_key.id)

  def test_transform_compiled_computation_returns_compiled_computation_with_id(
      self):
    tuple_type = computation_types.TensorType(tf.int32)
    compiled_computation = building_block_factory.create_compiled_identity(
        tuple_type)
    add_ids = compiled_computation_transforms.AddUniqueIDs()
    with self.subTest('first_comp_non_zero_id'):
      first_transformed_comp, mutated = add_ids.transform(compiled_computation)
      self.assertTrue(mutated)
      self.assertIsInstance(first_transformed_comp,
                            building_blocks.CompiledComputation)
      self.assertTrue(
          first_transformed_comp.proto.tensorflow.HasField('cache_key'))
      self.assertNotEqual(first_transformed_comp.proto.tensorflow.cache_key.id,
                          0)
    with self.subTest('second_comp_same_id'):
      second_transformed_comp, mutated = add_ids.transform(compiled_computation)
      self.assertTrue(mutated)
      self.assertIsInstance(second_transformed_comp,
                            building_blocks.CompiledComputation)
      self.assertTrue(
          second_transformed_comp.proto.tensorflow.HasField('cache_key'))
      self.assertNotEqual(second_transformed_comp.proto.tensorflow.cache_key.id,
                          0)
      self.assertEqual(first_transformed_comp.proto.tensorflow.cache_key.id,
                       second_transformed_comp.proto.tensorflow.cache_key.id)
    with self.subTest('restart_transformation_same_id'):
      # Test that the sequence ids are the same if we run a new compiler pass.
      # With compiler running inside the `invoke` call, we need to ensure
      # running different computations doesn't produce the same ids.
      add_ids = compiled_computation_transforms.AddUniqueIDs()
      third_transformed_comp, mutated = add_ids.transform(compiled_computation)
      self.assertTrue(mutated)
      self.assertTrue(
          third_transformed_comp.proto.tensorflow.HasField('cache_key'))
      self.assertNotEqual(third_transformed_comp.proto.tensorflow.cache_key.id,
                          0)
      self.assertEqual(first_transformed_comp.proto.tensorflow.cache_key.id,
                       third_transformed_comp.proto.tensorflow.cache_key.id)
    with self.subTest('different_computation_different_id'):
      different_compiled_computation = _create_compiled_computation(
          lambda x: x + tf.constant(1.0),
          computation_types.TensorType(tf.float32))
      different_transformed_comp, mutated = add_ids.transform(
          different_compiled_computation)
      self.assertTrue(mutated)
      self.assertTrue(
          different_transformed_comp.proto.tensorflow.HasField('cache_key'))
      self.assertNotEqual(
          different_transformed_comp.proto.tensorflow.cache_key.id, 0)
      self.assertNotEqual(
          first_transformed_comp.proto.tensorflow.cache_key.id,
          different_transformed_comp.proto.tensorflow.cache_key.id)


if __name__ == '__main__':
  test_case.main()
