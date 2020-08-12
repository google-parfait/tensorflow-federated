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
import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation_context


class FederatedComputationContextTest(absltest.TestCase):

  def test_invoke_returns_value_with_correct_type(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack)
    comp = computations.tf_computation(lambda: tf.constant(10))
    with context_stack_impl.context_stack.install(context):
      result = context.invoke(comp, None)
    self.assertIsInstance(result, value_base.Value)
    self.assertEqual(str(result.type_signature), 'int32')

  def test_construction_populates_name(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack)
    self.assertEqual(context.name, 'FEDERATED')

  def test_suggested_name_populates_name_attribute(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack, suggested_name='FOO')
    self.assertEqual(context.name, 'FOO')

  def test_child_name_doesnt_conflict(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack, suggested_name='FOO')
    self.assertEqual(context.name, 'FOO')
    context2 = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack, suggested_name='FOO', parent=context)
    self.assertEqual(context2.name, 'FOO_1')
    context3 = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack, suggested_name='FOO', parent=context2)
    self.assertEqual(context3.name, 'FOO_2')

  def test_parent_populated_correctly(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack)
    context2 = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack, parent=context)
    self.assertIs(context2.parent, context)

  def test_bind_single_computation_to_reference(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack)
    data = building_blocks.Data('x', tf.int32)
    ref = context.bind_computation_to_reference(data)
    symbol_bindings = context.symbol_bindings
    bound_symbol_name = symbol_bindings[0][0]

    self.assertIsInstance(ref, building_blocks.Reference)
    self.assertEqual(ref.type_signature, data.type_signature)
    self.assertLen(symbol_bindings, 1)
    self.assertEqual(bound_symbol_name, ref.name)

  def test_bind_two_computations_to_reference(self):
    context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack)
    data = building_blocks.Data('x', tf.int32)
    float_data = building_blocks.Data('x', tf.float32)
    ref1 = context.bind_computation_to_reference(data)
    ref2 = context.bind_computation_to_reference(float_data)
    symbol_bindings = context.symbol_bindings

    self.assertIsInstance(ref1, building_blocks.Reference)
    self.assertIsInstance(ref2, building_blocks.Reference)

    self.assertEqual(ref1.type_signature, data.type_signature)
    self.assertEqual(ref2.type_signature, float_data.type_signature)
    self.assertLen(symbol_bindings, 2)
    self.assertEqual(symbol_bindings[0][0], ref1.name)
    self.assertEqual(symbol_bindings[1][0], ref2.name)


if __name__ == '__main__':
  absltest.main()
