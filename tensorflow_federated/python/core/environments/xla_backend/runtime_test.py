# Copyright 2021, The TensorFlow Federated Authors.
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
import jax
from jax.lib import xla_client
import numpy as np

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.xla_backend import runtime
from tensorflow_federated.python.core.environments.xla_backend import xla_serialization
from tensorflow_federated.python.core.impl.types import computation_types


class RuntimeTest(absltest.TestCase):

  def test_normalize_tensor_representation_int32(self):
    result = runtime.normalize_tensor_representation(
        10, computation_types.TensorType(np.int32)
    )
    self.assertIsInstance(result, np.int32)
    self.assertEqual(result, 10)

  def test_normalize_tensor_representation_int32x2x3(self):
    result = runtime.normalize_tensor_representation(
        np.array(((1, 2), (3, 4), (5, 6)), dtype=np.int32),
        computation_types.TensorType(np.int32, (3, 2)),
    )
    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.dtype, np.int32)
    self.assertEqual(result.shape, (3, 2))
    self.assertEqual(list(result.flatten()), [1, 2, 3, 4, 5, 6])

  def test_computation_callable_return_one_number(self):
    builder = xla_client.XlaBuilder('comp')
    xla_client.ops.Constant(builder, np.int32(10))
    xla_comp = builder.build()
    comp_type = computation_types.FunctionType(None, np.int32)
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp, [], comp_type
    )
    backend = jax.lib.xla_bridge.get_backend()
    comp_callable = runtime.ComputationCallable(comp_pb, comp_type, backend)
    self.assertIsInstance(comp_callable, runtime.ComputationCallable)
    self.assertEqual(str(comp_callable.type_signature), '( -> int32)')
    result = comp_callable()
    self.assertEqual(result, 10)

  def test_computation_callable_add_two_numbers(self):
    builder = xla_client.XlaBuilder('comp')
    shape = xla_client.shape_from_pyval(np.array(0, dtype=np.int32))
    params = [xla_client.ops.Parameter(builder, i, shape) for i in range(2)]
    xla_client.ops.Add(params[0], params[1])
    xla_comp = builder.build()
    comp_type = computation_types.FunctionType((np.int32, np.int32), np.int32)
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp, [0, 1], comp_type
    )
    backend = jax.lib.xla_bridge.get_backend()
    comp_callable = runtime.ComputationCallable(comp_pb, comp_type, backend)
    self.assertIsInstance(comp_callable, runtime.ComputationCallable)
    self.assertEqual(
        str(comp_callable.type_signature), '(<int32,int32> -> int32)'
    )
    result = comp_callable(
        structure.Struct([(None, np.int32(2)), (None, np.int32(3))])
    )
    self.assertEqual(result, 5)


if __name__ == '__main__':
  absltest.main()
