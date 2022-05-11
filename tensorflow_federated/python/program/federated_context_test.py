# Copyright 2022, The TensorFlow Federated Authors.
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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import federated_context


class _TestFederatedContext(federated_context.FederatedContext):

  def invoke(self, comp: computation_base.Computation, arg: Any) -> Any:
    return None


class ContainsOnlyServerPlacedDataTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('struct_unnamed', computation_types.StructType([
          (None, computation_types.TensorType(tf.bool)),
          (None, computation_types.TensorType(tf.int32)),
          (None, computation_types.TensorType(tf.string)),
      ])),
      ('struct_named', computation_types.StructType([
          ('a', computation_types.TensorType(tf.bool)),
          ('b', computation_types.TensorType(tf.int32)),
          ('c', computation_types.TensorType(tf.string)),
      ])),
      ('struct_nested', computation_types.StructType([
          ('x', computation_types.StructType([
              ('a', computation_types.TensorType(tf.bool)),
              ('b', computation_types.TensorType(tf.int32)),
          ])),
          ('y', computation_types.StructType([
              ('c', computation_types.TensorType(tf.string)),
          ])),
      ])),
      ('federated_struct', computation_types.FederatedType(
          computation_types.StructType([
              ('a', computation_types.TensorType(tf.bool)),
              ('b', computation_types.TensorType(tf.int32)),
              ('c', computation_types.TensorType(tf.string)),
          ]),
          placements.SERVER)),
      ('federated_sequence', computation_types.FederatedType(
          computation_types.SequenceType(
              computation_types.TensorType(tf.int32)),
          placements.SERVER)),
      ('federated_tensor', computation_types.FederatedType(
          computation_types.TensorType(tf.int32),
          placements.SERVER)),
      ('sequence', computation_types.SequenceType(
          computation_types.TensorType(tf.int32))),
      ('tensor', computation_types.TensorType(tf.int32)),
  )
  # pyformat: enable
  def test_returns_true(self, type_signature):
    result = federated_context.contains_only_server_placed_data(type_signature)

    self.assertTrue(result)

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated', computation_types.FederatedType(
          computation_types.TensorType(tf.int32),
          placements.CLIENTS)),
      ('function', computation_types.FunctionType(
          computation_types.TensorType(tf.int32),
          computation_types.TensorType(tf.int32))),
      ('placement', computation_types.PlacementType()),
  )
  # pyformat: enable
  def test_returns_false(self, type_signature):
    result = federated_context.contains_only_server_placed_data(type_signature)

    self.assertFalse(result)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_type_signature(self, type_signature):
    with self.assertRaises(TypeError):
      federated_context.contains_only_server_placed_data(type_signature)


class CheckInFederatedContextTest(parameterized.TestCase):

  def test_does_not_raise_value_error(self):
    context = _TestFederatedContext()

    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()

    with context_stack_impl.context_stack.install(context):
      try:
        federated_context.check_in_federated_context()
      except TypeError:
        self.fail('Raised TypeError unexpectedly.')

    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()

  @parameterized.named_parameters(
      ('local_cpp_async',
       execution_contexts.create_local_async_python_execution_context()),
      ('local_cpp_sync',
       execution_contexts.create_local_python_execution_context()),
  )
  def test_raises_value_error_with_context(self, context):
    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()

    with context_stack_impl.context_stack.install(context):
      with self.assertRaises(ValueError):
        federated_context.check_in_federated_context()

    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()

  def test_raises_value_error_with_context_nested(self):
    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()

    context = _TestFederatedContext()
    with context_stack_impl.context_stack.install(context):
      try:
        federated_context.check_in_federated_context()
      except TypeError:
        self.fail('Raised TypeError unexpectedly.')

      context = execution_contexts.create_local_python_execution_context()
      with context_stack_impl.context_stack.install(context):
        with self.assertRaises(ValueError):
          federated_context.check_in_federated_context()

    with self.assertRaises(ValueError):
      federated_context.check_in_federated_context()


if __name__ == '__main__':
  absltest.main()
