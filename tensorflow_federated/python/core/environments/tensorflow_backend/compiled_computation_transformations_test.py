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

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.core.environments.tensorflow_backend import compiled_computation_transformations
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_computation_transformations
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_test_utils
from tensorflow_federated.python.core.impl.types import computation_types


def _create_compiled_computation(py_fn, parameter_type, layout_map=None):
  proto, type_signature = (
      tensorflow_computation_factory.create_computation_for_py_fn(
          py_fn, parameter_type, layout_map
      )
  )
  return building_blocks.CompiledComputation(
      proto, type_signature=type_signature
  )


class TensorFlowOptimizerTest(absltest.TestCase):

  def test_should_transform_compiled_computation(self):
    tuple_type = computation_types.TensorType(np.int32)
    compiled_proto, compiled_type = (
        tensorflow_computation_factory.create_identity(tuple_type)
    )
    compiled_computation = building_blocks.CompiledComputation(
        compiled_proto, name='a', type_signature=compiled_type
    )
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transformations.TensorFlowOptimizer(
        config
    )
    self.assertTrue(tf_optimizer.should_transform(compiled_computation))

  def test_should_not_transform_reference(self):
    reference = building_blocks.Reference('x', np.int32)
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transformations.TensorFlowOptimizer(
        config
    )
    self.assertFalse(tf_optimizer.should_transform(reference))

  def test_transform_compiled_computation_returns_compiled_computation(self):
    tuple_type = computation_types.TensorType(np.int32)
    proto, function_type = tensorflow_computation_factory.create_identity(
        tuple_type,
        layout_map=computation_pb2.TensorFlow.LayoutMap(
            name_to_sharding_spec={'v': 'unsharded'}
        ),
    )
    compiled_computation = building_blocks.CompiledComputation(
        proto, name=None, type_signature=function_type
    )

    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transformations.TensorFlowOptimizer(
        config
    )
    transformed_comp, mutated = tf_optimizer.transform(compiled_computation)
    self.assertTrue(mutated)
    self.assertIsInstance(transformed_comp, building_blocks.CompiledComputation)
    self.assertTrue(transformed_comp.proto.tensorflow.HasField('parameter'))
    self.assertFalse(transformed_comp.proto.tensorflow.initialize_op)
    self.assertEqual(
        transformed_comp.proto.tensorflow.layout_map.name_to_sharding_spec.get(
            'v'
        ),
        'unsharded',
    )

  def test_transform_compiled_computation_returns_compiled_computation_without_empty_fields(
      self,
  ):
    proto, type_signature = tensorflow_computation_factory.create_empty_tuple()
    compiled_computation = building_blocks.CompiledComputation(
        proto, type_signature=type_signature
    )
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transformations.TensorFlowOptimizer(
        config
    )
    transformed_comp, mutated = tf_optimizer.transform(compiled_computation)
    self.assertTrue(mutated)
    self.assertIsInstance(transformed_comp, building_blocks.CompiledComputation)
    self.assertFalse(transformed_comp.proto.tensorflow.HasField('parameter'))
    self.assertEmpty(
        transformed_comp.proto.tensorflow.layout_map.name_to_sharding_spec
    )
    self.assertFalse(transformed_comp.proto.tensorflow.initialize_op)

  def test_transform_compiled_computation_semantic_equivalence(self):
    tuple_type = computation_types.TensorType(np.int32)
    compiled_proto, compiled_type = (
        tensorflow_computation_factory.create_identity(tuple_type)
    )
    compiled_computation = building_blocks.CompiledComputation(
        compiled_proto, name='a', type_signature=compiled_type
    )
    config = tf.compat.v1.ConfigProto()
    tf_optimizer = compiled_computation_transformations.TensorFlowOptimizer(
        config
    )
    transformed_comp, mutated = tf_optimizer.transform(compiled_computation)
    self.assertTrue(mutated)
    self.assertIsInstance(transformed_comp, building_blocks.CompiledComputation)
    zero_before_transform = tensorflow_computation_test_utils.run_tensorflow(
        compiled_computation.proto, 0
    )
    zero_after_transform = tensorflow_computation_test_utils.run_tensorflow(
        transformed_comp.proto, 0
    )
    self.assertEqual(zero_before_transform, zero_after_transform)


class AddUniqueIDsTest(absltest.TestCase):

  def test_should_transform_compiled_tf_computation(self):
    tuple_type = computation_types.TensorType(np.int32)
    compiled_proto, compiled_type = (
        tensorflow_computation_factory.create_identity(tuple_type)
    )
    compiled_computation = building_blocks.CompiledComputation(
        compiled_proto, name='a', type_signature=compiled_type
    )
    self.assertTrue(
        compiled_computation_transformations.AddUniqueIDs().should_transform(
            compiled_computation
        )
    )

  def test_should_not_transform_non_compiled_computations(self):
    reference = building_blocks.Reference('x', np.int32)
    self.assertFalse(
        compiled_computation_transformations.AddUniqueIDs().should_transform(
            reference
        )
    )

  def test_transform_same_compiled_computation_different_type_signature(self):
    # First create no-arg computation that returns an empty tuple. This will
    # be compared against a omputation that returns a nested empty tuple, which
    # should produce a different ID.
    proto, type_signature = (
        tensorflow_computation_factory.create_unary_operator(
            lambda x: (), operand_type=computation_types.StructType([])
        )
    )
    empty_tuple_computation = building_blocks.CompiledComputation(
        proto, type_signature=type_signature
    )
    add_ids = compiled_computation_transformations.AddUniqueIDs()
    first_transformed_comp, mutated = add_ids.transform(empty_tuple_computation)
    self.assertTrue(mutated)
    self.assertIsInstance(
        first_transformed_comp, building_blocks.CompiledComputation
    )
    self.assertTrue(
        first_transformed_comp.proto.tensorflow.HasField('cache_key')
    )
    self.assertNotEqual(first_transformed_comp.proto.tensorflow.cache_key.id, 0)
    # Now create the same NoOp tf.Graph, but with a different binding and
    # type_signature.
    proto, type_signature = (
        tensorflow_computation_factory.create_unary_operator(
            lambda x: ((),), operand_type=computation_types.StructType([])
        )
    )
    nested_empty_tuple_computation = building_blocks.CompiledComputation(
        proto, type_signature=type_signature
    )
    second_transformed_comp, mutated = add_ids.transform(
        nested_empty_tuple_computation
    )
    self.assertTrue(mutated)
    self.assertIsInstance(
        second_transformed_comp, building_blocks.CompiledComputation
    )
    self.assertTrue(
        second_transformed_comp.proto.tensorflow.HasField('cache_key')
    )
    self.assertNotEqual(
        second_transformed_comp.proto.tensorflow.cache_key.id, 0
    )
    # Assert the IDs are different based on the type signture.
    self.assertNotEqual(
        first_transformed_comp.proto.tensorflow.cache_key.id,
        second_transformed_comp.proto.tensorflow.cache_key.id,
    )

  def test_transform_compiled_computation_returns_compiled_computation_with_id(
      self,
  ):
    tuple_type = computation_types.TensorType(np.int32)
    compiled_proto, compiled_type = (
        tensorflow_computation_factory.create_identity(tuple_type)
    )
    compiled_computation = building_blocks.CompiledComputation(
        compiled_proto, name='a', type_signature=compiled_type
    )
    add_ids = compiled_computation_transformations.AddUniqueIDs()
    with self.subTest('first_comp_non_zero_id'):
      first_transformed_comp, mutated = add_ids.transform(compiled_computation)
      self.assertTrue(mutated)
      self.assertIsInstance(
          first_transformed_comp, building_blocks.CompiledComputation
      )
      self.assertTrue(
          first_transformed_comp.proto.tensorflow.HasField('cache_key')
      )
      self.assertNotEqual(
          first_transformed_comp.proto.tensorflow.cache_key.id, 0
      )
    with self.subTest('second_comp_same_id'):
      second_transformed_comp, mutated = add_ids.transform(compiled_computation)
      self.assertTrue(mutated)
      self.assertIsInstance(
          second_transformed_comp, building_blocks.CompiledComputation
      )
      self.assertTrue(
          second_transformed_comp.proto.tensorflow.HasField('cache_key')
      )
      self.assertNotEqual(
          second_transformed_comp.proto.tensorflow.cache_key.id, 0
      )
      self.assertEqual(
          first_transformed_comp.proto.tensorflow.cache_key.id,
          second_transformed_comp.proto.tensorflow.cache_key.id,
      )
    with self.subTest('restart_transformation_same_id'):
      # Test that the sequence ids are the same if we run a new compiler pass.
      # With compiler running inside the `invoke` call, we need to ensure
      # running different computations doesn't produce the same ids.
      add_ids = compiled_computation_transformations.AddUniqueIDs()
      third_transformed_comp, mutated = add_ids.transform(compiled_computation)
      self.assertTrue(mutated)
      self.assertTrue(
          third_transformed_comp.proto.tensorflow.HasField('cache_key')
      )
      self.assertNotEqual(
          third_transformed_comp.proto.tensorflow.cache_key.id, 0
      )
      self.assertEqual(
          first_transformed_comp.proto.tensorflow.cache_key.id,
          third_transformed_comp.proto.tensorflow.cache_key.id,
      )
    with self.subTest('different_computation_different_id'):
      different_compiled_computation = _create_compiled_computation(
          lambda x: x + np.float32(1.0),
          computation_types.TensorType(np.float32),
      )
      different_transformed_comp, mutated = add_ids.transform(
          different_compiled_computation
      )
      self.assertTrue(mutated)
      self.assertTrue(
          different_transformed_comp.proto.tensorflow.HasField('cache_key')
      )
      self.assertNotEqual(
          different_transformed_comp.proto.tensorflow.cache_key.id, 0
      )
      self.assertNotEqual(
          first_transformed_comp.proto.tensorflow.cache_key.id,
          different_transformed_comp.proto.tensorflow.cache_key.id,
      )


class VerifyAllowedOpsTest(absltest.TestCase):

  def test_should_transform_tf_computation(self):
    tuple_type = computation_types.TensorType(np.int32)
    compiled_proto, compiled_type = (
        tensorflow_computation_factory.create_identity(tuple_type)
    )
    compiled_computation = building_blocks.CompiledComputation(
        compiled_proto, name='a', type_signature=compiled_type
    )
    self.assertTrue(
        compiled_computation_transformations.VerifyAllowedOps(
            frozenset()
        ).should_transform(compiled_computation)
    )

  def test_should_not_transform_non_compiled_computations(self):
    reference = building_blocks.Reference('x', np.int32)
    self.assertFalse(
        compiled_computation_transformations.VerifyAllowedOps(
            frozenset()
        ).should_transform(reference)
    )

  def test_transform_only_allowed_ops(self):
    tuple_type = computation_types.TensorType(np.int32)
    compiled_proto, compiled_type = (
        tensorflow_computation_factory.create_identity(tuple_type)
    )
    compiled_computation = building_blocks.CompiledComputation(
        compiled_proto, name='a', type_signature=compiled_type
    )
    allowed_op_names = frozenset(
        ['Const', 'PartitionedCall', 'Identity', 'Placeholder']
    )
    _, mutated = compiled_computation_transformations.VerifyAllowedOps(
        allowed_op_names
    ).transform(compiled_computation)
    self.assertFalse(mutated)

  def test_transform_disallowed_ops(self):
    tuple_type = computation_types.TensorType(np.int32)
    compiled_proto, compiled_type = (
        tensorflow_computation_factory.create_identity(tuple_type)
    )
    compiled_computation = building_blocks.CompiledComputation(
        compiled_proto, name='a', type_signature=compiled_type
    )
    allowed_op_names = frozenset(['Identity'])
    with self.assertRaises(
        tensorflow_computation_transformations.DisallowedOpInTensorFlowComputationError
    ):
      compiled_computation_transformations.VerifyAllowedOps(
          allowed_op_names
      ).transform(compiled_computation)


class RaiseOnDisallowedOpTest(absltest.TestCase):

  def test_should_transform_tf_computation(self):
    tuple_type = computation_types.TensorType(np.int32)
    compiled_proto, compiled_type = (
        tensorflow_computation_factory.create_identity(tuple_type)
    )
    compiled_computation = building_blocks.CompiledComputation(
        compiled_proto, name='a', type_signature=compiled_type
    )
    self.assertTrue(
        compiled_computation_transformations.RaiseOnDisallowedOp(
            frozenset()
        ).should_transform(compiled_computation)
    )

  def test_should_not_transform_non_compiled_computations(self):
    reference = building_blocks.Reference('x', np.int32)
    self.assertFalse(
        compiled_computation_transformations.RaiseOnDisallowedOp(
            frozenset()
        ).should_transform(reference)
    )

  def test_transform_no_disallowed_ops(self):
    tuple_type = computation_types.TensorType(np.int32)
    compiled_proto, compiled_type = (
        tensorflow_computation_factory.create_identity(tuple_type)
    )
    compiled_computation = building_blocks.CompiledComputation(
        compiled_proto, name='a', type_signature=compiled_type
    )
    disallowed_op_names = frozenset(['ShardedFilename'])
    _, mutated = compiled_computation_transformations.RaiseOnDisallowedOp(
        disallowed_op_names
    ).transform(compiled_computation)
    self.assertFalse(mutated)

  def test_transform_disallowed_ops(self):
    tuple_type = computation_types.TensorType(np.int32)
    compiled_proto, compiled_type = (
        tensorflow_computation_factory.create_identity(tuple_type)
    )
    compiled_computation = building_blocks.CompiledComputation(
        compiled_proto, name='a', type_signature=compiled_type
    )
    disallowed_op_names = frozenset(['Identity'])
    with self.assertRaises(
        tensorflow_computation_transformations.DisallowedOpInTensorFlowComputationError
    ):
      compiled_computation_transformations.RaiseOnDisallowedOp(
          disallowed_op_names
      ).transform(compiled_computation)


if __name__ == '__main__':
  absltest.main()
