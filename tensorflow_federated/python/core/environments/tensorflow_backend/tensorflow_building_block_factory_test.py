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

from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_test_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class CreateGenericConstantTest(absltest.TestCase):

  def test_raises_on_none_type(self):
    with self.assertRaises(TypeError):
      tensorflow_building_block_factory.create_generic_constant(None, 0)

  def test_raises_non_scalar(self):
    with self.assertRaises(TypeError):
      tensorflow_building_block_factory.create_generic_constant([np.int32], [0])

  def test_constructs_tensor_zero(self):
    tensor_type = computation_types.TensorType(np.float32, [2, 2])
    tensor_zero = tensorflow_building_block_factory.create_generic_constant(
        tensor_type, 0
    )
    self.assertEqual(tensor_zero.type_signature, tensor_type)
    self.assertIsInstance(tensor_zero, building_blocks.Call)
    self.assertTrue(
        np.array_equal(
            tensorflow_computation_test_utils.run_tensorflow(
                tensor_zero.function.proto
            ),
            np.zeros([2, 2]),
        )
    )

  def test_create_unnamed_tuple_zero(self):
    tensor_type = computation_types.TensorType(np.float32, [2, 2])
    tuple_type = computation_types.StructType((tensor_type, tensor_type))
    tuple_zero = tensorflow_building_block_factory.create_generic_constant(
        tuple_type, 0
    )
    self.assertEqual(tuple_zero.type_signature, tuple_type)
    self.assertIsInstance(tuple_zero, building_blocks.Call)
    result = tensorflow_computation_test_utils.run_tensorflow(
        tuple_zero.function.proto
    )
    self.assertLen(result, 2)
    self.assertTrue(np.array_equal(result[0], np.zeros([2, 2])))
    self.assertTrue(np.array_equal(result[1], np.zeros([2, 2])))

  def test_create_named_tuple_one(self):
    tensor_type = computation_types.TensorType(np.float32, [2, 2])
    tuple_type = computation_types.StructType(
        [('a', tensor_type), ('b', tensor_type)]
    )

    tuple_zero = tensorflow_building_block_factory.create_generic_constant(
        tuple_type, 1
    )

    self.assertEqual(tuple_zero.type_signature, tuple_type)
    self.assertIsInstance(tuple_zero, building_blocks.Call)
    result = tensorflow_computation_test_utils.run_tensorflow(
        tuple_zero.function.proto
    )
    self.assertLen(result, 2)
    self.assertTrue(np.array_equal(result.a, np.ones([2, 2])))
    self.assertTrue(np.array_equal(result.b, np.ones([2, 2])))

  def test_create_federated_tensor_one(self):
    fed_type = computation_types.FederatedType(
        computation_types.TensorType(np.float32, [2, 2]), placements.CLIENTS
    )
    fed_zero = tensorflow_building_block_factory.create_generic_constant(
        fed_type, 1
    )
    self.assertEqual(fed_zero.type_signature.member, fed_type.member)
    self.assertEqual(fed_zero.type_signature.placement, fed_type.placement)
    self.assertTrue(fed_zero.type_signature.all_equal)
    self.assertIsInstance(fed_zero, building_blocks.Call)
    self.assertIsInstance(fed_zero.function, building_blocks.Intrinsic)
    self.assertEqual(
        fed_zero.function.uri, intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri
    )
    self.assertIsInstance(fed_zero.argument, building_blocks.Call)
    self.assertTrue(
        np.array_equal(
            tensorflow_computation_test_utils.run_tensorflow(
                fed_zero.argument.function.proto
            ),
            np.ones([2, 2]),
        )
    )

  def test_create_federated_named_tuple_one(self):
    tuple_type = [
        ('a', computation_types.TensorType(np.float32, [2, 2])),
        ('b', computation_types.TensorType(np.float32, [2, 2])),
    ]
    fed_type = computation_types.FederatedType(tuple_type, placements.SERVER)
    fed_zero = tensorflow_building_block_factory.create_generic_constant(
        fed_type, 1
    )
    self.assertEqual(fed_zero.type_signature.member, fed_type.member)
    self.assertEqual(fed_zero.type_signature.placement, fed_type.placement)
    self.assertTrue(fed_zero.type_signature.all_equal)
    self.assertIsInstance(fed_zero, building_blocks.Call)
    self.assertIsInstance(fed_zero.function, building_blocks.Intrinsic)
    self.assertEqual(
        fed_zero.function.uri, intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri
    )
    self.assertIsInstance(fed_zero.argument, building_blocks.Call)
    result = tensorflow_computation_test_utils.run_tensorflow(
        fed_zero.argument.function.proto
    )
    self.assertLen(result, 2)
    self.assertTrue(np.array_equal(result.a, np.ones([2, 2])))
    self.assertTrue(np.array_equal(result.b, np.ones([2, 2])))

  def test_create_named_tuple_of_federated_tensors_zero(self):
    fed_type = computation_types.FederatedType(
        computation_types.TensorType(np.float32, [2, 2]),
        placements.CLIENTS,
        all_equal=True,
    )
    tuple_type = computation_types.StructType(
        [('a', fed_type), ('b', fed_type)]
    )

    zero = tensorflow_building_block_factory.create_generic_constant(
        tuple_type, 0
    )

    fed_zero = zero.argument[0]
    self.assertEqual(zero.type_signature, tuple_type)
    self.assertIsInstance(fed_zero.function, building_blocks.Intrinsic)
    self.assertEqual(
        fed_zero.function.uri, intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri
    )
    self.assertIsInstance(fed_zero.argument, building_blocks.Call)
    actual_result = tensorflow_computation_test_utils.run_tensorflow(
        fed_zero.argument.function.proto
    )
    self.assertTrue(np.array_equal(actual_result, np.zeros([2, 2])))


if __name__ == '__main__':
  absltest.main()
