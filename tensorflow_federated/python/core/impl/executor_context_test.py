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
"""Tests for executor_context.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_context
from tensorflow_federated.python.core.impl import tensorflow_deserialization
from tensorflow_federated.python.core.impl import transformations


class _FakeExecutor(executor_base.Executor):
  """Just returns computations submitted for execution for testing."""

  def execute(self, computation_proto):
    return computation_proto


class ExecutorContextTest(absltest.TestCase):

  def test_something(self):
    context = executor_context.ExecutorContext(_FakeExecutor())
    add_ten = computations.tf_computation(lambda x: x + 10, tf.int32)
    computation_proto = context.invoke(add_ten, 20)
    # TODO(b/113116813): Use reference executor when available instead of the
    # manual execution below.
    comp = transformations.name_compiled_computations(
        computation_building_blocks.ComputationBuildingBlock.from_proto(
            computation_proto))
    self.assertEqual(str(comp), 'comp#1(comp#2())')
    graph = tf.Graph()
    arg = tensorflow_deserialization.deserialize_and_call_tf_computation(
        comp.argument.function.proto, None, graph)
    result = tensorflow_deserialization.deserialize_and_call_tf_computation(
        comp.function.proto, arg, graph)
    session = tf.Session(graph=graph)
    result_val = session.run(result)
    self.assertEqual(result_val, 30)


if __name__ == '__main__':
  absltest.main()
