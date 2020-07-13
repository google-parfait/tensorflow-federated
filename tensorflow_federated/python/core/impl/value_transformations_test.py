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

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl import value_transformations
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import test_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import placement_literals


def _count_intrinsics(comp, uri):

  def _predicate(comp):
    return (isinstance(comp, building_blocks.Intrinsic) and uri is not None and
            comp.uri == uri)

  return tree_analysis.count(comp, _predicate)


class ReplaceIntrinsicsWithCallableTest(absltest.TestCase):

  def test_raises_type_error_with_none_comp(self):
    uri = 'intrinsic'
    body = lambda x: x

    with self.assertRaises(TypeError):
      value_transformations.replace_intrinsics_with_callable(
          None, uri, body, context_stack_impl.context_stack)

  def test_raises_type_error_with_none_uri(self):
    comp = test_utils.create_lambda_to_dummy_called_intrinsic(
        parameter_name='a')
    body = lambda x: x

    with self.assertRaises(TypeError):
      value_transformations.replace_intrinsics_with_callable(
          comp, None, body, context_stack_impl.context_stack)

  def test_raises_type_error_with_none_body(self):
    comp = test_utils.create_lambda_to_dummy_called_intrinsic(
        parameter_name='a')
    uri = 'intrinsic'

    with self.assertRaises(TypeError):
      value_transformations.replace_intrinsics_with_callable(
          comp, uri, None, context_stack_impl.context_stack)

  def test_raises_type_error_with_none_context_stack(self):
    comp = test_utils.create_lambda_to_dummy_called_intrinsic(
        parameter_name='a')
    uri = 'intrinsic'
    body = lambda x: x

    with self.assertRaises(TypeError):
      value_transformations.replace_intrinsics_with_callable(
          comp, uri, body, None)

  def test_replaces_intrinsic(self):
    comp = test_utils.create_lambda_to_dummy_called_intrinsic(
        parameter_name='a')
    uri = 'intrinsic'
    body = lambda x: x

    transformed_comp, modified = value_transformations.replace_intrinsics_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(comp.compact_representation(), '(a -> intrinsic(a))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(a -> (intrinsic_arg -> intrinsic_arg)(a))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replaces_nested_intrinsic(self):
    fn = test_utils.create_lambda_to_dummy_called_intrinsic(parameter_name='a')
    block = test_utils.create_dummy_block(fn, variable_name='b')
    comp = block
    uri = 'intrinsic'
    body = lambda x: x

    transformed_comp, modified = value_transformations.replace_intrinsics_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(comp.compact_representation(),
                     '(let b=data in (a -> intrinsic(a)))')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let b=data in (a -> (intrinsic_arg -> intrinsic_arg)(a)))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replaces_chained_intrinsics(self):
    fn = test_utils.create_lambda_to_dummy_called_intrinsic(parameter_name='a')
    arg = building_blocks.Data('data', tf.int32)
    call = test_utils.create_chained_calls([fn, fn], arg)
    comp = call
    uri = 'intrinsic'
    body = lambda x: x

    transformed_comp, modified = value_transformations.replace_intrinsics_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(comp.compact_representation(),
                     '(a -> intrinsic(a))((a -> intrinsic(a))(data))')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(a -> (intrinsic_arg -> intrinsic_arg)(a))((a -> (intrinsic_arg -> intrinsic_arg)(a))(data))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_replace_other_intrinsic(self):
    comp = test_utils.create_lambda_to_dummy_called_intrinsic(
        parameter_name='a')
    uri = 'other'
    body = lambda x: x

    transformed_comp, modified = value_transformations.replace_intrinsics_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(transformed_comp.compact_representation(),
                     '(a -> intrinsic(a))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class ReplaceIntrinsicsWithBodiesTest(absltest.TestCase):

  def test_raises_on_none(self):
    context_stack = context_stack_impl.context_stack
    with self.assertRaises(TypeError):
      value_transformations.replace_intrinsics_with_bodies(None, context_stack)

  def test_federated_weighted_mean_reduces(self):
    uri = intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri
    context_stack = context_stack_impl.context_stack

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placement_literals.CLIENTS))
    def foo(x):
      return intrinsics.federated_mean(x, x)

    foo_building_block = building_blocks.ComputationBuildingBlock.from_proto(
        foo._computation_proto)
    count_before_reduction = _count_intrinsics(foo_building_block, uri)
    reduced, modified = value_transformations.replace_intrinsics_with_bodies(
        foo_building_block, context_stack)
    count_after_reduction = _count_intrinsics(reduced, uri)
    self.assertGreater(count_before_reduction, 0)
    self.assertEqual(count_after_reduction, 0)
    self.assertTrue(modified)

  def test_federated_sum_reduces(self):
    uri = intrinsic_defs.FEDERATED_SUM.uri
    context_stack = context_stack_impl.context_stack

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placement_literals.CLIENTS))
    def foo(x):
      return intrinsics.federated_sum(x)

    foo_building_block = building_blocks.ComputationBuildingBlock.from_proto(
        foo._computation_proto)

    count_before_reduction = _count_intrinsics(foo_building_block, uri)
    reduced, modified = value_transformations.replace_intrinsics_with_bodies(
        foo_building_block, context_stack)
    count_after_reduction = _count_intrinsics(reduced, uri)
    self.assertGreater(count_before_reduction, 0)
    self.assertEqual(count_after_reduction, 0)
    self.assertTrue(modified)

  def test_generic_divide_reduces(self):
    uri = intrinsic_defs.GENERIC_DIVIDE.uri
    context_stack = context_stack_impl.context_stack
    comp = building_blocks.Intrinsic(
        uri, computation_types.FunctionType([tf.float32, tf.float32],
                                            tf.float32))

    count_before_reduction = _count_intrinsics(comp, uri)
    reduced, modified = value_transformations.replace_intrinsics_with_bodies(
        comp, context_stack)
    count_after_reduction = _count_intrinsics(reduced, uri)

    self.assertGreater(count_before_reduction, 0)
    self.assertEqual(count_after_reduction, 0)
    tree_analysis.check_intrinsics_whitelisted_for_reduction(reduced)
    self.assertTrue(modified)


if __name__ == '__main__':
  absltest.main()
