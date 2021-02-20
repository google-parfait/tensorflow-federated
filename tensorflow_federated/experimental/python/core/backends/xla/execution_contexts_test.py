# Copyright 2020, The TensorFlow Federated Authors.
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
from jax.lib.xla_bridge import xla_client
import numpy as np

from tensorflow_federated.experimental.python.core.backends.xla import execution_contexts
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.utils import xla_serialization


class ExecutionContextsTest(absltest.TestCase):

  def test_create_local_execution_context(self):
    context = execution_contexts.create_local_execution_context()
    self.assertIsInstance(context, execution_context.ExecutionContext)

  def test_set_local_execution_context(self):
    builder = xla_client.XlaBuilder('comp')
    xla_client.ops.Parameter(builder, 0, xla_client.shape_from_pyval(tuple()))
    xla_client.ops.Constant(builder, np.int32(10))
    xla_comp = builder.build()
    comp_type = computation_types.FunctionType(None, np.int32)
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp, [], comp_type)
    ctx_stack = context_stack_impl.context_stack
    comp = computation_impl.ComputationImpl(comp_pb, ctx_stack)
    execution_contexts.set_local_execution_context()
    self.assertEqual(comp(), 10)


if __name__ == '__main__':
  absltest.main()
