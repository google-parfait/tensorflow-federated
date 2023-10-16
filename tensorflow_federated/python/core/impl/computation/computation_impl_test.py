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
import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import type_test_utils


class ConcreteComputationTest(absltest.TestCase):

  def test_something(self):
    # TODO: b/113112108 - Revise these tests after a more complete
    # implementation is in place.

    # At the moment, this should succeed, as both the computation body and the
    # type are well-formed.
    computation_impl.ConcreteComputation(
        pb.Computation(
            **{
                'type': type_serialization.serialize_type(
                    computation_types.FunctionType(np.int32, np.int32)
                ),
                'intrinsic': pb.Intrinsic(uri='whatever'),
            }
        ),
        context_stack_impl.context_stack,
    )

    # This should fail, as the proto is not well-formed.
    self.assertRaises(
        NotImplementedError,
        computation_impl.ConcreteComputation,
        pb.Computation(),
        context_stack_impl.context_stack,
    )

    # This should fail, as "10" is not an instance of pb.Computation.
    self.assertRaises(
        TypeError,
        computation_impl.ConcreteComputation,
        10,
        context_stack_impl.context_stack,
    )

  def test_with_type_preserves_python_container(self):
    struct_return_type = computation_types.FunctionType(
        np.int32, computation_types.StructType([(None, np.int32)])
    )
    original_comp = computation_impl.ConcreteComputation(
        pb.Computation(
            **{
                'type': type_serialization.serialize_type(struct_return_type),
                'intrinsic': pb.Intrinsic(uri='whatever'),
            }
        ),
        context_stack_impl.context_stack,
    )

    list_return_type = computation_types.FunctionType(
        np.int32,
        computation_types.StructWithPythonType([(None, np.int32)], list),
    )
    fn_with_annotated_type = computation_impl.ConcreteComputation.with_type(
        original_comp, list_return_type
    )
    type_test_utils.assert_types_identical(
        list_return_type, fn_with_annotated_type.type_signature
    )

  def test_with_type_raises_non_assignable_type(self):
    int_return_type = computation_types.FunctionType(np.int32, np.int32)
    original_comp = computation_impl.ConcreteComputation(
        pb.Computation(
            **{
                'type': type_serialization.serialize_type(int_return_type),
                'intrinsic': pb.Intrinsic(uri='whatever'),
            }
        ),
        context_stack_impl.context_stack,
    )

    list_return_type = computation_types.FunctionType(
        np.int32,
        computation_types.StructWithPythonType([(None, np.int32)], list),
    )
    with self.assertRaises(computation_types.TypeNotAssignableError):
      computation_impl.ConcreteComputation.with_type(
          original_comp, list_return_type
      )


class FromBuildingBlockTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      computation_impl.ConcreteComputation.from_building_block(None)

  def test_converts_building_block_to_computation(self):
    buiding_block = building_blocks.Lambda(
        'x', np.int32, building_blocks.Reference('x', np.int32)
    )
    computation = computation_impl.ConcreteComputation.from_building_block(
        buiding_block
    )
    self.assertIsInstance(computation, computation_impl.ConcreteComputation)


if __name__ == '__main__':
  absltest.main()
