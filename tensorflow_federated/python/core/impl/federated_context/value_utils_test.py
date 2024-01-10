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
import numpy as np

from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import computation_factory
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation_context
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.federated_context import value_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class ValueUtilsTest(parameterized.TestCase):

  def run(self, result=None):
    fc_context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack
    )
    with context_stack_impl.context_stack.install(fc_context):
      super().run(result)

  def test_get_curried(self):
    type_sec = computation_types.StructType([
        computation_types.TensorType(np.int32),
        computation_types.TensorType(np.int32),
    ])
    computation_proto = computation_factory.create_lambda_identity(type_sec)
    type_signature = computation_types.FunctionType(type_sec, type_sec)
    building_block = building_blocks.CompiledComputation(
        proto=computation_proto, name='test', type_signature=type_signature
    )
    value = value_impl.Value(building_block)

    curried = value_utils.get_curried(value)

    self.assertEqual(
        curried.type_signature.compact_representation(),
        '(int32 -> (int32 -> <int32,int32>))',
    )
    self.assertEqual(
        curried.comp.compact_representation(),
        '(arg0 -> (arg1 -> comp#test(<arg0,arg1>)))',
    )

  def test_ensure_federated_value(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def _(x):
      x = value_impl.to_value(x, type_spec=None)
      value_utils.ensure_federated_value(x, placements.CLIENTS)
      return x

  def test_ensure_federated_value_wrong_placement(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def _(x):
      x = value_impl.to_value(x, type_spec=None)
      with self.assertRaises(TypeError):
        value_utils.ensure_federated_value(x, placements.SERVER)
      return x

  def test_ensure_federated_value_implicitly_zippable(self):

    @federated_computation.federated_computation(
        computation_types.StructType((
            computation_types.FederatedType(np.int32, placements.CLIENTS),
            computation_types.FederatedType(np.int32, placements.CLIENTS),
        ))
    )
    def _(x):
      x = value_impl.to_value(x, type_spec=None)
      value_utils.ensure_federated_value(x)
      return x

  def test_ensure_federated_value_fails_on_unzippable(self):

    @federated_computation.federated_computation(
        computation_types.StructType((
            computation_types.FederatedType(np.int32, placements.CLIENTS),
            computation_types.FederatedType(np.int32, placements.SERVER),
        ))
    )
    def _(x):
      x = value_impl.to_value(x, type_spec=None)
      with self.assertRaises(TypeError):
        value_utils.ensure_federated_value(x)
      return x


if __name__ == '__main__':
  absltest.main()
