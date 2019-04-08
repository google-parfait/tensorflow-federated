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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import six
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import transformation_utils


def _to_building_block(comp):
  """Converts a computation into a computation building block.

  Args:
    comp: An instance of `computation_impl.ComputationImpl`.

  Returns:
    A instance of `computation_building_blocks.ComputationBuildingBlock`
    representing the `computation_impl.ComputationImpl`.
  """
  py_typecheck.check_type(comp, computation_impl.ComputationImpl)
  proto = computation_impl.ComputationImpl.get_proto(comp)
  return computation_building_blocks.ComputationBuildingBlock.from_proto(proto)


class TrivialSubclass(transformation_utils.BoundVariableTracker):

  def update(self, comp):
    pass

  def __str__(self):
    return 'TrivialSubclass'

  def __eq__(self, other):
    return id(self) == id(other)


class TransformationsTest(absltest.TestCase):

  def test_transform_postorder_fails_on_none(self):

    def transform(comp):
      return comp

    with self.assertRaises(TypeError):
      transformation_utils.transform_postorder(None, transform)

  def test_transform_postorder_with_lambda_call_selection_and_reference(self):

    @computations.federated_computation(
        [computation_types.FunctionType(tf.int32, tf.int32), tf.int32])
    def foo(f, x):
      return f(x)

    comp = _to_building_block(foo)
    self.assertEqual(str(comp), '(foo_arg -> foo_arg[0](foo_arg[1]))')

    def _transformation_fn_generator():
      n = 0
      while True:
        n = n + 1

        def _fn(x):
          return computation_building_blocks.Call(
              computation_building_blocks.Intrinsic(
                  'F{}'.format(n),
                  computation_types.FunctionType(x.type_signature,
                                                 x.type_signature)), x)

        yield _fn

    transformation_fn_sequence = _transformation_fn_generator()
    # pylint: disable=unnecessary-lambda
    tx_fn = lambda x: six.next(transformation_fn_sequence)(x)
    # pylint: enable=unnecessary-lambda
    transfomed_comp = transformation_utils.transform_postorder(comp, tx_fn)
    self.assertEqual(
        str(transfomed_comp),
        'F6((foo_arg -> F5(F2(F1(foo_arg)[0])(F4(F3(foo_arg)[1])))))')

  # TODO(b/113123410): Add more tests for corner cases of `transform_preorder`.

  def test_trivial_subclass_init_fails_bad_args(self):
    with self.assertRaises(TypeError):
      TrivialSubclass()
    with self.assertRaises(TypeError):
      TrivialSubclass(0, None)
    with self.assertRaises(TypeError):
      TrivialSubclass('x', 0)

  def test_trivial_subclass_init(self):
    x = TrivialSubclass('x', None)
    self.assertEqual(x.name, 'x')
    self.assertIsNone(x.value)

  def test_sequential_binding_node_fails_bad_args(self):
    with self.assertRaises(TypeError):
      transformation_utils.SequentialBindingNode(None)
    with self.assertRaises(TypeError):
      transformation_utils.SequentialBindingNode(0)

  def test_sequential_binding_node_initialization(self):

    trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialSubclass('trivial_name', None))

    self.assertEqual(trivial_instance.payload.name, 'trivial_name')
    self.assertEmpty(trivial_instance.children)
    self.assertIsNone(trivial_instance.payload.value)
    self.assertIsNone(trivial_instance.parent)
    self.assertIsNone(trivial_instance.younger_sibling)
    self.assertIsNone(trivial_instance.older_sibling)

  def test_comptracker_trivial_subclass_init_bad_args(self):
    with self.assertRaises(TypeError):
      TrivialSubclass(0, None)
    with self.assertRaises(TypeError):
      TrivialSubclass('x', 0)

  def test_comptracker_trivial_subclass_parent_child_relationship(self):
    trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialSubclass('trivial_name', None))
    second_trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialSubclass('second_trivial_name', None))

    self.assertNotEqual(trivial_instance, second_trivial_instance)
    second_trivial_instance.set_parent(trivial_instance)
    trivial_instance.add_child(0, second_trivial_instance)
    self.assertEqual(trivial_instance.get_child(0), second_trivial_instance)
    self.assertIsNone(trivial_instance.get_child(1))
    self.assertEqual(second_trivial_instance.parent, trivial_instance)
    with self.assertRaises(TypeError):
      trivial_instance.set_parent(0)
    with self.assertRaises(TypeError):
      second_trivial_instance.add_child(0, 0)

  def test_comptracker_trivial_subclass_sibling_relationship(self):
    trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialSubclass('trivial_name', None))
    second_trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialSubclass('second_trivial_name', None))

    self.assertNotEqual(trivial_instance, second_trivial_instance)
    trivial_instance.set_younger_sibling(second_trivial_instance)
    self.assertEqual(trivial_instance.younger_sibling, second_trivial_instance)
    second_trivial_instance.set_older_sibling(trivial_instance)
    self.assertEqual(second_trivial_instance.older_sibling, trivial_instance)
    with self.assertRaises(TypeError):
      trivial_instance.set_younger_sibling(0)
    with self.assertRaises(TypeError):
      second_trivial_instance.set_older_sibling(0)

  def test_comptracker_trivial_subclass_cousin_relationship(self):
    trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialSubclass('trivial_name', None))
    second_trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialSubclass('second_trivial_name', None))
    third_trivial_instance = transformation_utils.SequentialBindingNode(
        TrivialSubclass('third_trivial_name', None))
    trivial_instance.add_child(0, second_trivial_instance)
    trivial_instance.add_child(1, third_trivial_instance)
    second_trivial_instance.set_parent(trivial_instance)
    third_trivial_instance.set_parent(trivial_instance)
    second_trivial_instance_relations = [
        second_trivial_instance.parent, second_trivial_instance.older_sibling,
        second_trivial_instance.younger_sibling
    ] + list(six.itervalues(second_trivial_instance.children))

    third_trivial_instance_relations = [
        third_trivial_instance.parent, third_trivial_instance.older_sibling,
        third_trivial_instance.younger_sibling
    ] + list(six.itervalues(third_trivial_instance.children))
    self.assertNotIn(second_trivial_instance, third_trivial_instance_relations)
    self.assertNotIn(third_trivial_instance, second_trivial_instance_relations)
    self.assertEqual(
        id(second_trivial_instance.parent), id(third_trivial_instance.parent))

  def test_outer_context_pointer_equality(self):
    outer_context = transformation_utils.OuterContextPointer()
    other_outer_context = transformation_utils.OuterContextPointer()
    self.assertNotEqual(id(outer_context), id(other_outer_context))
    self.assertEqual(str(outer_context), 'OuterContext')
    self.assertEqual(outer_context, other_outer_context)

  def test_outer_context_pointer_cant_update(self):
    outer_context = transformation_utils.OuterContextPointer()
    with self.assertRaises(RuntimeError):
      outer_context.update()

  def test_simple_block_snapshot(self):
    used1 = computation_building_blocks.Reference('used1', tf.int32)
    used2 = computation_building_blocks.Reference('used2', tf.int32)
    ref = computation_building_blocks.Reference('x', used1.type_signature)
    lower_block = computation_building_blocks.Block([('x', used1)], ref)
    higher_block = computation_building_blocks.Block([('used1', used2)],
                                                     lower_block)
    self.assertEqual(
        str(higher_block), '(let used1=used2 in (let x=used1 in x))')
    snapshot = transformation_utils.scope_count_snapshot(higher_block)
    self.assertEqual(snapshot[str(lower_block)]['x'], 1)
    self.assertEqual(snapshot[str(higher_block)]['used1'], 1)
    self.assertIsNone(snapshot[str(higher_block)].get('x'))

  def test_scope_snapshot_block_overwrite(self):
    innermost = computation_building_blocks.Reference('x', tf.int32)
    intermediate_arg = computation_building_blocks.Reference('y', tf.int32)
    item2 = computation_building_blocks.Block([('x', intermediate_arg)],
                                              innermost)
    item1 = computation_building_blocks.Reference('x', tf.int32)
    mediate_tuple = computation_building_blocks.Tuple([item1, item2])
    used = computation_building_blocks.Reference('used', tf.int32)
    used1 = computation_building_blocks.Reference('used1', tf.int32)
    outer_block = computation_building_blocks.Block([('x', used), ('y', used1)],
                                                    mediate_tuple)
    self.assertEqual(
        str(outer_block), '(let x=used,y=used1 in <x,(let x=y in x)>)')
    snapshot = transformation_utils.scope_count_snapshot(outer_block)
    self.assertEqual(snapshot[str(item2)], {'x': 1})
    self.assertEqual(snapshot[str(outer_block)], {'x': 1, 'y': 1})
    self.assertIsNone(snapshot.get(str(mediate_tuple)))

  def test_scope_snapshot_lambda_overwrite(self):
    inner_x = computation_building_blocks.Reference('x', tf.int32)
    inner_lambda = computation_building_blocks.Lambda('x', tf.int32, inner_x)
    outer_x = computation_building_blocks.Reference('x', tf.int32)
    call = computation_building_blocks.Call(inner_lambda, outer_x)
    outer_lambda = computation_building_blocks.Lambda('x', tf.int32, call)
    snapshot = transformation_utils.scope_count_snapshot(outer_lambda)
    self.assertEqual(snapshot[str(inner_lambda)], {'x': 1})
    self.assertEqual(snapshot[str(outer_lambda)], {'x': 1})
    outer_call = computation_building_blocks.Call(inner_lambda, outer_x)
    third_lambda = computation_building_blocks.Lambda('x', tf.int32, outer_call)
    second_snapshot = transformation_utils.scope_count_snapshot(third_lambda)
    self.assertEqual(str(third_lambda), '(x -> (x -> x)(x))')
    self.assertEqual(second_snapshot[str(inner_lambda)], {'x': 1})
    self.assertEqual(second_snapshot[str(outer_lambda)], {'x': 1})
    self.assertEqual(second_snapshot[str(third_lambda)], {'x': 1})

  def test_nested_lambda_block_overwrite_scope_snapshot(self):
    innermost_x = computation_building_blocks.Reference('x', tf.int32)
    inner_lambda = computation_building_blocks.Lambda('x', tf.int32,
                                                      innermost_x)
    second_x = computation_building_blocks.Reference('x', tf.int32)
    called_lambda = computation_building_blocks.Call(inner_lambda, second_x)
    block_input = computation_building_blocks.Reference('block_in', tf.int32)
    lower_block = computation_building_blocks.Block([('x', block_input)],
                                                    called_lambda)
    second_lambda = computation_building_blocks.Lambda('block_in', tf.int32,
                                                       lower_block)
    third_x = computation_building_blocks.Reference('x', tf.int32)
    second_call = computation_building_blocks.Call(second_lambda, third_x)
    final_input = computation_building_blocks.Data('test_data', tf.int32)
    last_block = computation_building_blocks.Block([('x', final_input)],
                                                   second_call)
    global_snapshot = transformation_utils.scope_count_snapshot(last_block)
    self.assertEqual(
        str(last_block),
        '(let x=test_data in (block_in -> (let x=block_in in (x -> x)(x)))(x))')
    self.assertLen(global_snapshot, 4)
    self.assertEqual(global_snapshot[str(inner_lambda)], {'x': 1})
    self.assertEqual(global_snapshot[str(lower_block)], {'x': 1})
    self.assertEqual(global_snapshot[str(second_lambda)], {'block_in': 1})
    self.assertEqual(global_snapshot[str(last_block)], {'x': 1})

  def test_scope_snapshot_called_lambdas(self):
    comp = computation_building_blocks.Tuple(
        [computation_building_blocks.Data('test', tf.int32)])
    input1 = computation_building_blocks.Reference('input1',
                                                   comp.type_signature)
    first_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input1', input1.type_signature,
                                           input1), comp)
    input2 = computation_building_blocks.Reference(
        'input2', first_level_call.type_signature)
    second_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input2', input2.type_signature,
                                           input2), first_level_call)
    self.assertEqual(
        str(second_level_call),
        '(input2 -> input2)((input1 -> input1)(<test>))')
    global_snapshot = transformation_utils.scope_count_snapshot(
        second_level_call)
    self.assertEqual(global_snapshot, {
        '(input2 -> input2)': {
            'input2': 1
        },
        '(input1 -> input1)': {
            'input1': 1
        }
    })


if __name__ == '__main__':
  absltest.main()
