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
from absl.testing import parameterized
import six
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import tensorflow_serialization
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


def _construct_complex_symbol_tree():
  """Constructs complex context tree for mutation testing."""
  symbol_tree = transformation_utils.SymbolTree(FakeTracker)
  for _ in range(2):
    symbol_tree._add_younger_sibling(fake_tracker_node_factory())
    symbol_tree._move_to_younger_sibling()
  symbol_tree._add_child(0, fake_tracker_node_factory())
  symbol_tree._move_to_child(0)
  for _ in range(2):
    symbol_tree._add_younger_sibling(fake_tracker_node_factory())
    symbol_tree._move_to_younger_sibling()
  symbol_tree._add_child(1, fake_tracker_node_factory())
  symbol_tree._move_to_child(1)
  for k in range(2):
    symbol_tree.move_to_parent_context()
    symbol_tree._move_to_older_sibling()
    symbol_tree._move_to_older_sibling()
    symbol_tree._add_child(k + 2, fake_tracker_node_factory())
  symbol_tree._move_to_child(3)
  return symbol_tree


class UpdatableTracker(transformation_utils.BoundVariableTracker):

  def __init__(self, name, value):
    super(UpdatableTracker, self).__init__(name, value)
    self.count = 0

  def update(self, comp):
    self.count += 1

  def __str__(self):
    return '{Count: ' + str(self.count) + '}'

  def __eq__(self, other):
    return id(self) == id(other)


class FakeTracker(transformation_utils.BoundVariableTracker):

  def update(self, comp=None):
    pass

  def __str__(self):
    return self.name

  def __eq__(self, other):
    return isinstance(other, FakeTracker)


def fake_tracker_node_factory():
  return transformation_utils.SequentialBindingNode(
      FakeTracker('FakeTracker', None))


class TrivialSubclass(transformation_utils.BoundVariableTracker):

  def update(self, comp):
    pass

  def __str__(self):
    return 'TrivialSubclass'

  def __eq__(self, other):
    return id(self) == id(other)


def _construct_trivial_instance_of_all_computation_building_blocks():
  cbb_list = []
  ref_to_x = computation_building_blocks.Reference('x', tf.int32)
  cbb_list.append(('reference', ref_to_x))
  lam = computation_building_blocks.Lambda('x', tf.int32, ref_to_x)
  cbb_list.append(('lambda', lam))
  block = computation_building_blocks.Block([('x', ref_to_x)], lam)
  cbb_list.append(('block', block))
  data = computation_building_blocks.Data('x', tf.int32)
  cbb_list.append(('data', data))
  function_type = computation_types.FunctionType(tf.int32, tf.int32)
  intrinsic = computation_building_blocks.Intrinsic('dummy_intrinsic',
                                                    function_type)
  cbb_list.append(('intrinsic', intrinsic))
  tff_tuple = computation_building_blocks.Tuple([ref_to_x])
  cbb_list.append(('tuple', tff_tuple))
  selection = computation_building_blocks.Selection(tff_tuple, index=0)
  cbb_list.append(('selection', selection))
  call = computation_building_blocks.Call(lam, ref_to_x)
  cbb_list.append(('call', call))
  fn = lambda: tf.constant(1)
  compiled_comp = computation_building_blocks.CompiledComputation(
      tensorflow_serialization.serialize_py_fn_as_tf_computation(
          fn, None, context_stack_impl.context_stack))
  cbb_list.append(('compiled_comp', compiled_comp))
  placement = computation_building_blocks.Placement(placement_literals.CLIENTS)
  cbb_list.append(('placement', placement))
  return cbb_list


def _construct_nested_tree():
  r"""Constructs computation with explicit ordering for testing traversals.

  The goal of this computation is to exercise each switch
  in transform_postorder_with_symbol_bindings, at least all those that recurse.

  The computation this function constructs can be represented as below.

  Notice that the body of the Lambda *does not depend on the Lambda's
  parameter*, so that if we were actually executing this call the argument will
  be thrown away.

                            Call
                           /    \
                 Lambda('arg')   Data('k')
                     |
                   Block('y','z')-------------
                  /                          |
  ['y'=Data('a'),'z'=Data('b')]              |
                                           Tuple
                                         /       \
                                   Block('v')     Block('x')-------
                                     / \              |            |
                       ['v'=Selection]   Data('g') ['x'=Data('h']  |
                             |                                     |
                             |                                     |
                             |                                 Block('w')
                             |                                   /   \
                           Tuple ------            ['w'=Data('i']     Data('j')
                         /              \
                 Block('t')             Block('u')
                  /     \              /          \
    ['t'=Data('c')]    Data('d') ['u'=Data('e')]  Data('f')


  If we are reading Data URIs, results of a postorder traversal should be:
  [a, b, c, d, e, f, g, h, i, j, k]

  If we are reading locals declarations, results of a postorder traversal should
  be:
  [t, u, v, w, x, y, z]

  And if we are reading both in an interleaved fashion, results of a postorder
  traversal should be:
  [a, b, c, d, t, e, f, u, g, v, h, i, j, w, x, y, z, k]

  Since we are also exposing the ability to hook into variable declarations,
  it is worthwhile considering the order in which variables are assigned in
  this tree. Notice that this order maps neither to preorder nor to postorder
  when purely considering the nodes of the tree above. This would be:
  [arg, y, z, t, u, v, x, w]

  Returns:
    An instance of `computation_building_blocks.ComputationBuildingBlock`
    satisfying the description above.
  """
  data_c = computation_building_blocks.Data('c', tf.float32)
  data_d = computation_building_blocks.Data('d', tf.float32)
  left_most_leaf = computation_building_blocks.Block([('t', data_c)], data_d)

  data_e = computation_building_blocks.Data('e', tf.float32)
  data_f = computation_building_blocks.Data('f', tf.float32)
  center_leaf = computation_building_blocks.Block([('u', data_e)], data_f)
  inner_tuple = computation_building_blocks.Tuple([left_most_leaf, center_leaf])

  selected = computation_building_blocks.Selection(inner_tuple, index=0)
  data_g = computation_building_blocks.Data('g', tf.float32)
  middle_block = computation_building_blocks.Block([('v', selected)], data_g)

  data_i = computation_building_blocks.Data('i', tf.float32)
  data_j = computation_building_blocks.Data('j', tf.float32)
  right_most_endpoint = computation_building_blocks.Block([('w', data_i)],
                                                          data_j)

  data_h = computation_building_blocks.Data('h', tf.int32)
  right_child = computation_building_blocks.Block([('x', data_h)],
                                                  right_most_endpoint)

  result = computation_building_blocks.Tuple([middle_block, right_child])
  data_a = computation_building_blocks.Data('a', tf.float32)
  data_b = computation_building_blocks.Data('b', tf.float32)
  dummy_outer_block = computation_building_blocks.Block([('y', data_a),
                                                         ('z', data_b)], result)
  dummy_lambda = computation_building_blocks.Lambda('arg', tf.float32,
                                                    dummy_outer_block)
  dummy_arg = computation_building_blocks.Data('k', tf.float32)
  called_lambda = computation_building_blocks.Call(dummy_lambda, dummy_arg)

  return called_lambda


def _get_number_of_nodes_via_transform_postorder(comp, predicate=None):
  """Returns the number of nodes in `comp` matching `predicate`."""
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  count = [0]  # TODO(b/129791812): Cleanup Python 2 and 3 compatibility.

  def fn(comp):
    if predicate is None or predicate(comp):
      count[0] += 1
    return comp

  transformation_utils.transform_postorder(comp, fn)
  return count[0]


def _get_number_of_nodes_via_transform_postorder_with_symbol_bindings(
    comp, predicate=None):
  """Returns the number of nodes in `comp` matching `predicate`."""
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
  count = [0]  # TODO(b/129791812): Cleanup Python 2 and 3 compatibility.

  def fn(comp, ctxt_tree):
    del ctxt_tree
    if predicate is None or predicate(comp):
      count[0] += 1
    return comp

  transformation_utils.transform_postorder_with_symbol_bindings(
      comp, fn, empty_context_tree)

  return count[0]


class TransformationUtilsTest(parameterized.TestCase):

  def test_transform_postorder_fails_on_none_comp(self):

    def transform(comp):
      return comp

    with self.assertRaises(TypeError):
      transformation_utils.transform_postorder(None, transform)

  def test_transform_postorder_fails_on_none_transform(self):
    comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      transformation_utils.transform_postorder(comp, None)

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

  @parameterized.named_parameters(
      _construct_trivial_instance_of_all_computation_building_blocks() +
      [('complex_tree', _construct_nested_tree())])
  def test_transform_postorder_returns_untransformed(self, comp):

    def transform_noop(internal_comp):
      return internal_comp

    same_comp = transformation_utils.transform_postorder(comp, transform_noop)
    self.assertEqual(same_comp.tff_repr, comp.tff_repr)

  @parameterized.named_parameters(
      _construct_trivial_instance_of_all_computation_building_blocks())
  def test_transform_postorder_constructs_new_internal(self, comp):

    def transform_noop(comp):
      return comp

    same_comp = transformation_utils.transform_postorder(comp, transform_noop)

    if not isinstance(comp, (computation_building_blocks.CompiledComputation,
                             computation_building_blocks.Data,
                             computation_building_blocks.Intrinsic,
                             computation_building_blocks.Placement,
                             computation_building_blocks.Reference)):
      self.assertNotEqual(id(comp), id(same_comp))

  def test_transform_postorder_hits_all_nodes_once(self):
    complex_ast = _construct_nested_tree()
    self.assertEqual(
        _get_number_of_nodes_via_transform_postorder(complex_ast), 22)

  def test_transform_postorder_walks_to_leaves_in_postorder(self):
    complex_ast = _construct_nested_tree()

    leaf_name_order = []

    def transform(comp):
      if isinstance(comp, computation_building_blocks.Data):
        leaf_name_order.append(comp.uri)
      return comp

    transformation_utils.transform_postorder(complex_ast, transform)

    self.assertEqual(leaf_name_order,
                     ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'])

  def test_transform_postorder_walks_block_locals_postorder(self):
    complex_ast = _construct_nested_tree()

    leaf_name_order = []

    def transform(comp):
      if isinstance(comp, computation_building_blocks.Block):
        for name, _ in comp.locals:
          leaf_name_order.append(name)
      return comp

    transformation_utils.transform_postorder(complex_ast, transform)

    self.assertEqual(leaf_name_order, ['t', 'u', 'v', 'w', 'x', 'y', 'z'])

  def test_transform_postorder_walks_through_all_internal_nodes_postorder(self):
    """Checks `transform_postorder` walks correctly through any internal node.

    This test is split from the one above because it tests extra cases
    in `transform_postorder`; in particular, all instances of
    `computation_building_blocks.ComputationBuildingBlock` which kick off
    recursive calls of `transform_postorder` are exercised in this test,
    while only a subset are exercised in the above. For example, if the
    logic ingesting a `Call` breaks, this test will fail and the one above
    may pass.
    """
    complex_ast = _construct_nested_tree()

    leaf_name_order = []

    def transform(comp):
      if isinstance(comp, computation_building_blocks.Block):
        for name, _ in comp.locals:
          leaf_name_order.append(name)
      elif isinstance(comp, computation_building_blocks.Data):
        leaf_name_order.append(comp.uri)
      return comp

    transformation_utils.transform_postorder(complex_ast, transform)
    postorder_nodes = [
        'a', 'b', 'c', 'd', 't', 'e', 'f', 'u', 'g', 'v', 'h', 'i', 'j', 'w',
        'x', 'y', 'z', 'k'
    ]

    self.assertEqual(leaf_name_order, list(postorder_nodes))

  # TODO(b/113123410): Add more tests for corner cases of `transform_preorder`.

  def test_transform_postorder_with_symbol_bindings_fails_on_none_comp(self):
    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)

    def transform(comp, ctxt_tree):
      del ctxt_tree
      return comp

    with self.assertRaises(TypeError):
      transformation_utils.transform_postorder_with_symbol_bindings(
          None, transform, empty_context_tree)

  def test_transform_postorder_with_symbol_bindings_fails_on_none_transform(
      self):
    empty_symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    dummy_comp = computation_building_blocks.Reference('x', tf.int32)

    with self.assertRaises(TypeError):
      transformation_utils.transform_postorder_with_symbol_bindings(
          dummy_comp, None, empty_symbol_tree)

  def test_transform_postorder_with_symbol_bindings_fails_on_none_symbol_tree(
      self):
    dummy_comp = computation_building_blocks.Reference('x', tf.int32)

    def transform(comp, ctxt_tree):
      del ctxt_tree
      return comp

    with self.assertRaises(TypeError):
      transformation_utils.transform_postorder_with_symbol_bindings(
          dummy_comp, transform, None)

  @parameterized.named_parameters(
      _construct_trivial_instance_of_all_computation_building_blocks() +
      [('complex_ast', _construct_nested_tree())])
  def test_transform_postorder_with_symbol_bindings_returns_untransformed(
      self, comp):

    def transform_noop(comp, ctxt_tree):
      del ctxt_tree
      return comp

    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
    same_comp = transformation_utils.transform_postorder_with_symbol_bindings(
        comp, transform_noop, empty_context_tree)
    self.assertEqual(same_comp.tff_repr, comp.tff_repr)

  @parameterized.named_parameters(
      _construct_trivial_instance_of_all_computation_building_blocks())
  def test_transform_postorder_with_symbol_bindings_constructs_new_internal_nodes(
      self, comp):

    def transform_noop(comp, ctxt_tree):
      del ctxt_tree
      return comp

    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
    same_comp = transformation_utils.transform_postorder_with_symbol_bindings(
        comp, transform_noop, empty_context_tree)
    if not isinstance(comp, (computation_building_blocks.CompiledComputation,
                             computation_building_blocks.Data,
                             computation_building_blocks.Intrinsic,
                             computation_building_blocks.Placement,
                             computation_building_blocks.Reference)):
      self.assertNotEqual(id(comp), id(same_comp))

  def test_transform_postorder_with_symbol_bindings_hits_all_nodes_once(self):
    complex_ast = _construct_nested_tree()

    simple_count = _get_number_of_nodes_via_transform_postorder(complex_ast)
    with_hooks_count = _get_number_of_nodes_via_transform_postorder_with_symbol_bindings(
        complex_ast)

    self.assertEqual(with_hooks_count, simple_count)

  @parameterized.named_parameters(
      ('reference', computation_building_blocks.Reference),
      ('lambda', computation_building_blocks.Lambda),
      ('block', computation_building_blocks.Block),
      ('data', computation_building_blocks.Data),
      ('intrinsic', computation_building_blocks.Intrinsic),
      ('tuple', computation_building_blocks.Tuple),
      ('selection', computation_building_blocks.Selection),
      ('call', computation_building_blocks.Call),
      ('compiled_computation', computation_building_blocks.CompiledComputation),
      ('placement', computation_building_blocks.Placement))
  def test_transform_postorder_with_symbol_bindings_counts_each_type_correctly(
      self, cbb_type):
    complex_ast = _construct_nested_tree()

    simple_count = _get_number_of_nodes_via_transform_postorder(
        complex_ast, predicate=lambda x: isinstance(x, cbb_type))
    with_hooks_count = _get_number_of_nodes_via_transform_postorder_with_symbol_bindings(
        complex_ast, predicate=lambda x: isinstance(x, cbb_type))

    self.assertEqual(with_hooks_count, simple_count)

  def test_transform_postorder_hooks_walks_leaves_in_postorder(self):
    leaf_order = []
    outer_comp = _construct_nested_tree()

    def transform(comp, ctxt_tree):
      del ctxt_tree
      if isinstance(comp, computation_building_blocks.Data):
        leaf_order.append(comp.uri)
      return comp

    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
    transformation_utils.transform_postorder_with_symbol_bindings(
        outer_comp, transform, empty_context_tree)
    self.assertEqual(leaf_order,
                     ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'])

  def test_transform_postorder_hooks_walks_block_locals_postorder(self):
    block_locals_order = []
    outer_comp = _construct_nested_tree()

    def transform(comp, ctxt_tree):
      del ctxt_tree
      if isinstance(comp, computation_building_blocks.Block):
        for name, _ in comp.locals:
          block_locals_order.append(name)
      return comp

    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
    transformation_utils.transform_postorder_with_symbol_bindings(
        outer_comp, transform, empty_context_tree)
    self.assertEqual(block_locals_order, ['t', 'u', 'v', 'w', 'x', 'y', 'z'])

  def test_transform_postorder_hooks_walks_variable_declarations_in_order(self):
    variable_binding_order = []
    outer_comp = _construct_nested_tree()

    class PreorderHookTracker(transformation_utils.BoundVariableTracker):

      def __init__(self, name, value):
        variable_binding_order.append(name)
        super(PreorderHookTracker, self).__init__(name, value)

      def update(self, value):
        pass

      def __str__(self):
        pass

      def __eq__(self, other):
        return NotImplemented

    empty_context_tree = transformation_utils.SymbolTree(PreorderHookTracker)
    transformation_utils.transform_postorder_with_symbol_bindings(
        outer_comp, lambda x, y: x, empty_context_tree)
    self.assertEqual(variable_binding_order,
                     ['arg', 'y', 'z', 't', 'u', 'v', 'x', 'w'])

  def test_transform_postorder_hooks_walks_postorder_interleaved(self):
    named_node_order = []
    outer_comp = _construct_nested_tree()

    def transform(comp, ctxt_tree):
      del ctxt_tree
      if isinstance(comp, computation_building_blocks.Block):
        for name, _ in comp.locals:
          named_node_order.append(name)
      elif isinstance(comp, computation_building_blocks.Data):
        named_node_order.append(comp.uri)
      return comp

    empty_context_tree = transformation_utils.SymbolTree(FakeTracker)
    transformation_utils.transform_postorder_with_symbol_bindings(
        outer_comp, transform, empty_context_tree)
    correct_results = [
        'a', 'b', 'c', 'd', 't', 'e', 'f', 'u', 'g', 'v', 'h', 'i', 'j', 'w',
        'x', 'y', 'z', 'k'
    ]
    self.assertEqual(named_node_order, correct_results)

  def test_transform_postorder_with_symbol_bindings_binds_lambda_param(self):
    result = computation_building_blocks.Reference('x', tf.int32)
    lam = computation_building_blocks.Lambda('x', tf.int32, result)
    empty_symbol_tree = transformation_utils.SymbolTree(UpdatableTracker)
    value_holder = []

    def transform(comp, ctxt_tree):
      if isinstance(comp, computation_building_blocks.Reference):
        ctxt_tree.update_payload_tracking_reference(comp)
        value_holder.append(ctxt_tree.get_payload_with_name(comp.name))
      return comp

    _ = transformation_utils.transform_postorder_with_symbol_bindings(
        lam, transform, empty_symbol_tree)

    self.assertEqual(value_holder[0].count, 1)
    self.assertEqual(value_holder[0].name, 'x')
    self.assertEqual(value_holder[0].value, None)

  def test_transform_postorder_with_symbol_bindings_binds_single_block_local(
      self):
    result = computation_building_blocks.Reference('x', tf.int32)
    arg = computation_building_blocks.Data('input_data', tf.int32)
    block = computation_building_blocks.Block([('x', arg)], result)
    empty_symbol_tree = transformation_utils.SymbolTree(UpdatableTracker)
    value_holder = []

    def transform(comp, ctxt_tree):
      if isinstance(comp, computation_building_blocks.Reference):
        ctxt_tree.update_payload_tracking_reference(comp)
        value_holder.append(ctxt_tree.get_payload_with_name(comp.name))
      return comp

    _ = transformation_utils.transform_postorder_with_symbol_bindings(
        block, transform, empty_symbol_tree)

    self.assertEqual(value_holder[0].count, 1)
    self.assertEqual(value_holder[0].name, 'x')
    self.assertEqual(value_holder[0].value, arg)

  def test_transform_postorder_with_symbol_bindings_binds_sequential_block_locals(
      self):
    result = computation_building_blocks.Reference('x', tf.int32)
    arg = computation_building_blocks.Data('input_data', tf.int32)
    arg2 = computation_building_blocks.Reference('x', tf.int32)
    block = computation_building_blocks.Block([('x', arg), ('x', arg2)], result)
    empty_symbol_tree = transformation_utils.SymbolTree(UpdatableTracker)
    value_holder = []

    def transform(comp, ctxt_tree):
      if isinstance(comp, computation_building_blocks.Reference):
        ctxt_tree.update_payload_tracking_reference(comp)
        value_holder.append(ctxt_tree.get_payload_with_name(comp.name))
      return comp

    _ = transformation_utils.transform_postorder_with_symbol_bindings(
        block, transform, empty_symbol_tree)

    self.assertEqual(value_holder[0].count, 1)
    self.assertEqual(value_holder[0].name, 'x')
    self.assertEqual(value_holder[0].value, arg)
    self.assertEqual(value_holder[1].count, 1)
    self.assertEqual(value_holder[1].name, 'x')
    self.assertEqual(value_holder[1].value, arg2)

  def test_symbol_tree_initializes(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    self.assertIsInstance(symbol_tree.active_node.payload,
                          transformation_utils.OuterContextPointer)
    self.assertTrue(
        py_typecheck.check_subclass(symbol_tree.payload_type,
                                    transformation_utils.BoundVariableTracker))

  def test_symbol_tree_node_reuse_fails(self):

    fake_tracker_node_one = transformation_utils.SequentialBindingNode(
        FakeTracker('FakeTracker', None))
    fake_tracker_node_two = transformation_utils.SequentialBindingNode(
        FakeTracker('FakeTracker', None))
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_child(0, fake_tracker_node_one)
    symbol_tree._move_to_child(0)
    symbol_tree._add_younger_sibling(fake_tracker_node_two)
    symbol_tree._move_to_younger_sibling()
    with self.assertRaisesRegexp(ValueError, 'can only appear once'):
      symbol_tree._add_child(1, fake_tracker_node_one)
    with self.assertRaisesRegexp(ValueError, 'can only appear once'):
      symbol_tree._add_younger_sibling(fake_tracker_node_one)

  def test_bad_mock_class_fails_symbol_tree(self):

    class BadMock(object):
      pass

    with self.assertRaisesRegexp(TypeError, 'subclass'):
      transformation_utils.SymbolTree(BadMock)

  def test_symbol_tree_get_payload_resolves_child_parent_name_conflict(self):

    def _construct_symbol_tree():
      """Constructs a symbol tree of the form below.

                      Outer Context
                           |
                           V
                       x_tracker
                           |
                           V
                      x_tracker2*

      Returns:
        Returns this tree and the payloads used to construct it.
      """
      symbol_tree = transformation_utils.SymbolTree(FakeTracker)
      x_tracker = FakeTracker('x', None)
      symbol_tree._add_child(
          0, transformation_utils.SequentialBindingNode(x_tracker))
      symbol_tree._move_to_child(0)
      x_tracker2 = FakeTracker('x', None)
      symbol_tree._add_child(
          1, transformation_utils.SequentialBindingNode(x_tracker2))
      symbol_tree._move_to_child(1)
      return symbol_tree, x_tracker, x_tracker2

    symbol_tree, _, x_tracker2 = _construct_symbol_tree()
    self.assertEqual(id(symbol_tree.get_payload_with_name('x')), id(x_tracker2))

  def test_symbol_tree_get_payload_resolves_sibling_name_conflict(self):

    def _construct_symbol_tree():
      """Constructs a symbol tree of the form below.

                      Outer Context
                           |
                           V
                       x_tracker
                           |
                      x_tracker2*

      Returns:
        Returns this tree and the payloads used to construct it.
      """
      symbol_tree = transformation_utils.SymbolTree(FakeTracker)
      x_tracker = FakeTracker('x', None)
      symbol_tree._add_child(
          0, transformation_utils.SequentialBindingNode(x_tracker))
      symbol_tree._move_to_child(0)
      x_tracker2 = FakeTracker('x', None)
      symbol_tree._add_younger_sibling(
          transformation_utils.SequentialBindingNode(x_tracker2))
      symbol_tree._move_to_younger_sibling()
      return symbol_tree, x_tracker, x_tracker2

    symbol_tree, _, x_tracker2 = _construct_symbol_tree()
    self.assertEqual(id(symbol_tree.get_payload_with_name('x')), id(x_tracker2))

  def test_symbol_tree_get_payload_addresses_parent(self):

    def _construct_symbol_tree():
      """Constructs a symbol tree of the form below.

                      Outer Context
                           |
                           V
                       z_tracker
                           |
                           V
                       x_tracker*

      Returns:
        Returns this tree and the payloads used to construct it.
      """
      symbol_tree = transformation_utils.SymbolTree(FakeTracker)
      z_tracker = FakeTracker('z', None)
      symbol_tree._add_child(
          0, transformation_utils.SequentialBindingNode(z_tracker))
      symbol_tree._move_to_child(0)
      x_tracker = FakeTracker('x', None)
      symbol_tree._add_child(
          1, transformation_utils.SequentialBindingNode(x_tracker))
      symbol_tree._move_to_child(1)
      return symbol_tree, z_tracker, x_tracker

    symbol_tree, z_tracker, _ = _construct_symbol_tree()
    self.assertEqual(id(symbol_tree.get_payload_with_name('z')), id(z_tracker))

  def test_symbol_tree_updates_correct_node_across_siblings(self):

    def _construct_symbol_tree():
      r"""Builds symbol tree with the structure below.

                      Outer Context
                           |
                           V
                        x_tracker
                           |
                        elder_y
                           |
                        young_y*

      Returns:
        Returns this tree and the `SequentialBindingNode`s
        used to construct it.
      """
      x_tracker = transformation_utils.SequentialBindingNode(
          UpdatableTracker('x', None))
      elder_y = transformation_utils.SequentialBindingNode(
          UpdatableTracker('y', None))
      young_y = transformation_utils.SequentialBindingNode(
          UpdatableTracker('y', None))

      complex_symbol_tree = transformation_utils.SymbolTree(UpdatableTracker)
      complex_symbol_tree._add_child(4, x_tracker)
      complex_symbol_tree._move_to_child(4)
      complex_symbol_tree._add_younger_sibling(elder_y)
      complex_symbol_tree._move_to_younger_sibling()
      complex_symbol_tree._add_younger_sibling(young_y)
      complex_symbol_tree._move_to_younger_sibling()
      return complex_symbol_tree, x_tracker, elder_y, young_y

    (complex_symbol_tree, x_tracker, elder_y,
     young_y) = _construct_symbol_tree()
    complex_symbol_tree.update_payload_tracking_reference(
        computation_building_blocks.Reference('x', tf.int32))
    complex_symbol_tree.update_payload_tracking_reference(
        computation_building_blocks.Reference('y', tf.int32))
    self.assertEqual(x_tracker.payload.count, 1)
    self.assertEqual(young_y.payload.count, 1)
    self.assertEqual(complex_symbol_tree.get_payload_with_name('x').count, 1)
    self.assertEqual(complex_symbol_tree.get_payload_with_name('y').count, 1)
    self.assertEqual(elder_y.payload.count, 0)
    with self.assertRaises(NameError):
      complex_symbol_tree.get_payload_with_name('z')

  def test_symbol_tree_updates_correct_node_across_generations(self):

    def _construct_symbol_tree():
      r"""Builds symbol tree with the structure below.

                      Outer Context
                           |
                           V
                        x_tracker
                           |
                        elder_y
                       /      \
                      V        V
                  young_y*   misdirect_z

      Returns:
        Returns this tree and the `SequentialBindingNode`s
        used to construct it.
      """
      x_tracker = transformation_utils.SequentialBindingNode(
          UpdatableTracker('x', None))
      elder_y = transformation_utils.SequentialBindingNode(
          UpdatableTracker('y', None))
      young_y = transformation_utils.SequentialBindingNode(
          UpdatableTracker('y', None))
      misdirect_z = transformation_utils.SequentialBindingNode(
          UpdatableTracker('z', None))

      complex_symbol_tree = transformation_utils.SymbolTree(UpdatableTracker)
      complex_symbol_tree._add_child(4, x_tracker)
      complex_symbol_tree._move_to_child(4)
      complex_symbol_tree._add_younger_sibling(elder_y)
      complex_symbol_tree._move_to_younger_sibling()
      complex_symbol_tree._add_child(5, young_y)
      complex_symbol_tree._move_to_child(5)
      complex_symbol_tree._add_child(6, misdirect_z)
      return (complex_symbol_tree, x_tracker, elder_y, young_y, misdirect_z)

    (complex_symbol_tree, x_tracker, elder_y, young_y,
     misdirect_z) = _construct_symbol_tree()
    complex_symbol_tree.update_payload_tracking_reference(
        computation_building_blocks.Reference('x', tf.int32))
    complex_symbol_tree.update_payload_tracking_reference(
        computation_building_blocks.Reference('y', tf.int32))
    self.assertEqual(x_tracker.payload.count, 1)
    self.assertEqual(young_y.payload.count, 1)
    self.assertEqual(elder_y.payload.count, 0)
    self.assertEqual(complex_symbol_tree.get_payload_with_name('x').count, 1)
    self.assertEqual(complex_symbol_tree.get_payload_with_name('y').count, 1)
    with self.assertRaises(NameError):
      complex_symbol_tree.get_payload_with_name('z')
    complex_symbol_tree.move_to_parent_context()
    complex_symbol_tree.update_payload_tracking_reference(
        computation_building_blocks.Reference('y', tf.int32))
    complex_symbol_tree.update_payload_tracking_reference(
        computation_building_blocks.Reference('y', tf.int32))
    self.assertEqual(elder_y.payload.count, 2)
    self.assertEqual(complex_symbol_tree.get_payload_with_name('y').count, 2)
    self.assertEqual(misdirect_z.payload.count, 0)
    complex_symbol_tree._move_to_older_sibling()
    with self.assertRaises(NameError):
      complex_symbol_tree.get_payload_with_name('y')

  def test_typechecking_in_symbol_tree_resolve_methods(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    with self.assertRaises(TypeError):
      symbol_tree.get_payload_with_name(0)
    with self.assertRaises(TypeError):
      symbol_tree.update_payload_tracking_reference(
          computation_building_blocks.Data('x', tf.bool))
    with self.assertRaises(TypeError):
      symbol_tree.update_payload_tracking_reference(0)

  def test_symbol_tree_ingest_variable_binding_bad_enum_fails(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    with self.assertRaises(AttributeError):
      symbol_tree.ingest_variable_binding(
          'x', computation_building_blocks.Data('x', tf.int32),
          transformation_utils.MutationMode.COUSIN)

  def test_symbol_tree_ingest_variable_binding_bad_args_fails(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    with self.assertRaises(TypeError):
      symbol_tree.ingest_variable_binding(
          0, computation_building_blocks.Reference('x', tf.int32),
          transformation_utils.MutationMode.SIBLING)
    with self.assertRaises(TypeError):
      symbol_tree.ingest_variable_binding(
          'x', 0, transformation_utils.MutationMode.SIBLING)
    with self.assertRaises(TypeError):
      symbol_tree.ingest_variable_binding(
          'x', computation_building_blocks.Reference('x', tf.int32),
          transformation_utils.MutationMode.CHILD)
    with self.assertRaises(TypeError):
      symbol_tree.ingest_variable_binding(
          'x', computation_building_blocks.Reference('x', tf.int32),
          transformation_utils.MutationMode.CHILD, 'y')

  def test_ingest_variable_binding_child_mode_adds_node_to_empty_tree(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    shadow_symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree.ingest_variable_binding(
        'x', computation_building_blocks.Reference('x', tf.int32),
        transformation_utils.MutationMode.CHILD, 0)
    shadow_symbol_tree._add_child(
        0,
        transformation_utils.SequentialBindingNode(
            FakeTracker('FakeTracker',
                        computation_building_blocks.Reference('x', tf.int32))))
    self.assertEqual(symbol_tree, shadow_symbol_tree)

  def test_ingest_variable_binding_sibling_mode_adds_node_to_empty_tree(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    shadow_symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    payload_to_add = FakeTracker(
        'x', computation_building_blocks.Data('a', tf.int32))
    shadow_symbol_tree._add_younger_sibling(
        transformation_utils.SequentialBindingNode(payload_to_add))

    symbol_tree.ingest_variable_binding(
        payload_to_add.name, payload_to_add.value,
        transformation_utils.MutationMode.SIBLING)

    self.assertEqual(symbol_tree, shadow_symbol_tree)

  def test_ingest_variable_binding_sibling_mode_adds_node_to_nonempty_tree(
      self):
    symbol_tree = _construct_complex_symbol_tree()
    shadow_symbol_tree = _construct_complex_symbol_tree()
    payload_to_add = FakeTracker(
        'x', computation_building_blocks.Data('a', tf.int32))
    shadow_symbol_tree._add_younger_sibling(
        transformation_utils.SequentialBindingNode(payload_to_add))

    symbol_tree.ingest_variable_binding(
        'x', computation_building_blocks.Data('a', tf.int32),
        transformation_utils.MutationMode.SIBLING)

    self.assertEqual(symbol_tree, shadow_symbol_tree)

  def test_ingest_child_mode_overwrites_existing_node_with_same_name(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree.ingest_variable_binding(
        'y', computation_building_blocks.Data('b', tf.int32),
        transformation_utils.MutationMode.CHILD, 1)
    resolved_y = symbol_tree.get_payload_with_name('y')
    self.assertEqual(resolved_y.value.uri, 'b')
    self.assertEqual(str(resolved_y.value.type_signature), 'int32')
    symbol_tree.move_to_parent_context()
    symbol_tree.ingest_variable_binding(
        'y', computation_building_blocks.Data('d', tf.bool),
        transformation_utils.MutationMode.CHILD, 1)
    changed_y = symbol_tree.get_payload_with_name('y')
    self.assertEqual(changed_y.value.uri, 'd')
    self.assertEqual(str(changed_y.value.type_signature), 'bool')

  def test_ingest_child_mode_overwrite_leaves_unrelated_node_alone(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree.ingest_variable_binding(
        'x', computation_building_blocks.Data('a', tf.bool),
        transformation_utils.MutationMode.CHILD, 0)
    symbol_tree.ingest_variable_binding(
        'y', computation_building_blocks.Data('b', tf.int32),
        transformation_utils.MutationMode.CHILD, 1)
    resolved_x = symbol_tree.get_payload_with_name('x')
    self.assertEqual(resolved_x.value.uri, 'a')
    self.assertEqual(str(resolved_x.value.type_signature), 'bool')
    symbol_tree.move_to_parent_context()
    symbol_tree.ingest_variable_binding(
        'y', computation_building_blocks.Data('d', tf.bool),
        transformation_utils.MutationMode.CHILD, 1)
    same_x = symbol_tree.get_payload_with_name('x')
    self.assertEqual(same_x.value, resolved_x.value)

  def test_ingest_child_mode_raises_error_on_name_conflict(self):
    symbol_tree = _construct_complex_symbol_tree()
    symbol_tree.ingest_variable_binding(
        'x', computation_building_blocks.Data('a', tf.bool),
        transformation_utils.MutationMode.CHILD, 0)
    symbol_tree.ingest_variable_binding(
        'y', computation_building_blocks.Data('b', tf.int32),
        transformation_utils.MutationMode.CHILD, 1)
    symbol_tree.move_to_parent_context()
    with self.assertRaises(ValueError):
      symbol_tree.ingest_variable_binding(
          'z', computation_building_blocks.Data('c', tf.int32),
          transformation_utils.MutationMode.CHILD, 1)

  def test_ingest_sibling_mode_overwrites_existing_node_with_same_name(self):
    symbol_tree = _construct_complex_symbol_tree()
    symbol_tree.ingest_variable_binding(
        'y', computation_building_blocks.Data('b', tf.int32),
        transformation_utils.MutationMode.SIBLING)
    resolved_y = symbol_tree.get_payload_with_name('y')
    self.assertEqual(resolved_y.value.uri, 'b')
    self.assertEqual(str(resolved_y.value.type_signature), 'int32')
    symbol_tree._move_to_older_sibling()
    symbol_tree.ingest_variable_binding(
        'y', computation_building_blocks.Data('d', tf.float32),
        transformation_utils.MutationMode.SIBLING)
    updated_y = symbol_tree.get_payload_with_name('y')
    self.assertEqual(updated_y.value.uri, 'd')
    self.assertEqual(str(updated_y.value.type_signature), 'float32')

  def test_ingest_sibling_mode_overwrite_leaves_unrelated_node_alone(self):
    symbol_tree = _construct_complex_symbol_tree()
    symbol_tree.ingest_variable_binding(
        'x', computation_building_blocks.Data('a', tf.bool),
        transformation_utils.MutationMode.SIBLING)
    symbol_tree.ingest_variable_binding(
        'y', computation_building_blocks.Data('b', tf.int32),
        transformation_utils.MutationMode.SIBLING)
    resolved_x = symbol_tree.get_payload_with_name('x')
    self.assertEqual(resolved_x.value.uri, 'a')
    self.assertEqual(str(resolved_x.value.type_signature), 'bool')
    symbol_tree._move_to_older_sibling()
    symbol_tree.ingest_variable_binding(
        'y', computation_building_blocks.Data('d', tf.float32),
        transformation_utils.MutationMode.SIBLING)
    same_x = symbol_tree.get_payload_with_name('x')
    self.assertEqual(same_x.value, resolved_x.value)

  def test_ingest_sibling_mode_raises_error_on_name_conflict(self):
    symbol_tree = _construct_complex_symbol_tree()
    symbol_tree.ingest_variable_binding(
        'x', computation_building_blocks.Data('a', tf.bool),
        transformation_utils.MutationMode.SIBLING)
    symbol_tree.ingest_variable_binding(
        'y', computation_building_blocks.Data('b', tf.int32),
        transformation_utils.MutationMode.SIBLING)
    symbol_tree._move_to_older_sibling()
    with self.assertRaises(ValueError):
      symbol_tree.ingest_variable_binding(
          'z', computation_building_blocks.Data('c', tf.float32),
          transformation_utils.MutationMode.SIBLING)

  def test_symbol_tree_move_to_bad_parent_fails(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_child(
        0,
        transformation_utils.SequentialBindingNode(
            FakeTracker('FakeTracker', None)))
    with self.assertRaisesRegexp(ValueError, 'nonexistent parent'):
      symbol_tree.move_to_parent_context()

  def test_symbol_tree_move_to_good_parent_succeeds(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_child(
        0,
        transformation_utils.SequentialBindingNode(
            FakeTracker('FakeTracker', None)))
    symbol_tree._move_to_child(0)
    symbol_tree.move_to_parent_context()
    self.assertEqual(symbol_tree.active_node.payload,
                     transformation_utils.OuterContextPointer())

  def test_symbol_tree_add_sibling(self):
    fake_node = transformation_utils.SequentialBindingNode(
        FakeTracker('FakeTracker', None))
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_younger_sibling(fake_node)
    symbol_tree._move_to_younger_sibling()
    self.assertEqual(id(symbol_tree.active_node), id(fake_node))
    self.assertIsNone(symbol_tree.active_node.children.get(0))
    self.assertIsNone(symbol_tree.active_node.younger_sibling)
    symbol_tree._move_to_older_sibling()
    self.assertEqual(symbol_tree.active_node.payload,
                     transformation_utils.OuterContextPointer())
    self.assertIsNotNone(symbol_tree.active_node.younger_sibling)
    self.assertIsNone(symbol_tree.active_node.children.get(0))

  def test_symbol_tree_move_to_bad_older_sibling_fails(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_younger_sibling(fake_tracker_node_factory())
    with self.assertRaisesRegexp(ValueError, 'nonexistent older sibling'):
      symbol_tree._move_to_older_sibling()

  def test_symbol_tree_has_younger_sibling(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_younger_sibling(fake_tracker_node_factory())
    self.assertIsNotNone(symbol_tree.active_node.younger_sibling)

  def test_symbol_tree_move_to_bad_younger_sibling_fails(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_younger_sibling(fake_tracker_node_factory())
    symbol_tree._move_to_younger_sibling()
    with self.assertRaisesRegexp(ValueError, 'nonexistent younger sibling'):
      symbol_tree._move_to_younger_sibling()
    symbol_tree._move_to_older_sibling()

  def test_symbol_tree_add_child(self):
    fake_node = transformation_utils.SequentialBindingNode(
        FakeTracker('FakeTracker', None))
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_child(0, fake_node)
    symbol_tree._move_to_child(0)
    self.assertEqual(id(symbol_tree.active_node), id(fake_node))
    symbol_tree.move_to_parent_context()
    self.assertEqual(symbol_tree.active_node.payload,
                     transformation_utils.OuterContextPointer())

  def test_symbol_tree_move_to_bad_child_fails(self):
    fake_node = transformation_utils.SequentialBindingNode(
        FakeTracker('FakeTracker', None))
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    symbol_tree._add_child(0, fake_node)
    with self.assertRaises(ValueError):
      symbol_tree._move_to_child(1)

  def test_complicated_symbol_tree_equality(self):
    first_tree = _construct_complex_symbol_tree()
    second_tree = _construct_complex_symbol_tree()
    self.assertEqual(first_tree, second_tree)
    second_tree._add_child(
        10,
        transformation_utils.SequentialBindingNode(FakeTracker('alpha', None)))
    self.assertNotEqual(first_tree, second_tree)
    self.assertNotEqual(second_tree, first_tree)

  def test_complicated_symbol_tree_equality_independent_of_active_node(self):
    first_tree = _construct_complex_symbol_tree()
    second_tree = _construct_complex_symbol_tree()
    second_tree.move_to_parent_context()
    second_tree._move_to_younger_sibling()
    second_tree._move_to_younger_sibling()
    self.assertEqual(first_tree, second_tree)

  def test_complicated_symbol_tree_resolves_string_correctly(self):
    symbol_tree = transformation_utils.SymbolTree(FakeTracker)
    for _ in range(2):
      symbol_tree._add_younger_sibling(fake_tracker_node_factory())
      symbol_tree._move_to_younger_sibling()
    symbol_tree._add_child(0, fake_tracker_node_factory())
    symbol_tree._move_to_child(0)
    for _ in range(2):
      symbol_tree._add_younger_sibling(fake_tracker_node_factory())
      symbol_tree._move_to_younger_sibling()
    symbol_tree._add_child(1, fake_tracker_node_factory())
    symbol_tree._move_to_child(1)
    for k in range(2):
      symbol_tree.move_to_parent_context()
      symbol_tree._move_to_older_sibling()
      symbol_tree._move_to_older_sibling()
      symbol_tree._add_child(k + 2, fake_tracker_node_factory())
    symbol_tree._move_to_child(3)

    self.assertEqual(
        str(symbol_tree),
        '[OuterContext]->{([FakeTracker*])}-[FakeTracker]-[FakeTracker]->{('
        '[FakeTracker]->{([FakeTracker])}-[FakeTracker]-[FakeTracker]->{([FakeTracker])})}'
    )
    symbol_tree.move_to_parent_context()
    self.assertEqual(
        str(symbol_tree),
        '[OuterContext*]->{([FakeTracker])}-[FakeTracker]-[FakeTracker]->{('
        '[FakeTracker]->{([FakeTracker])}-[FakeTracker]-[FakeTracker]->{([FakeTracker])})}'
    )
    symbol_tree._move_to_younger_sibling()
    symbol_tree._move_to_younger_sibling()
    self.assertEqual(
        str(symbol_tree),
        '[OuterContext]->{([FakeTracker])}-[FakeTracker]-[FakeTracker*]->{('
        '[FakeTracker]->{([FakeTracker])}-[FakeTracker]-[FakeTracker]->{([FakeTracker])})}'
    )

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
