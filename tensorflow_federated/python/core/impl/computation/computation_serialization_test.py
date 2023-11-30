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

from tensorflow_federated.python.core.impl.compiler import computation_factory
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import computation_serialization
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import computation_types


class ComputationSerializationTest(absltest.TestCase):

  def test_serialize_deserialize_round_trip(self):
    type_spec = computation_types.TensorType(np.int32)
    proto = computation_factory.create_lambda_identity(type_spec)
    comp = computation_impl.ConcreteComputation(
        proto, context_stack_impl.context_stack
    )
    serialized_comp = computation_serialization.serialize_computation(comp)
    deserialize_comp = computation_serialization.deserialize_computation(
        serialized_comp
    )
    self.assertIsInstance(deserialize_comp, computation_base.Computation)
    self.assertEqual(deserialize_comp, comp)


if __name__ == '__main__':
  absltest.main()
