# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""A library of static analysis functions that can be applied to ASTs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.compiler import transformation_utils


def count_types(comp, types):
  return count(comp, lambda x: isinstance(x, types))


def count(comp, predicate=None):
  """Returns the number of computations in `comp` matching `predicate`.

  Args:
    comp: The computation to test.
    predicate: An optional Python function that takes a computation as a
      parameter and returns a boolean value. If `None`, all computations are
      counted.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  counter = [0]

  def _function(comp):
    if predicate is None or predicate(comp):
      counter[0] += 1
    return comp, False

  transformation_utils.transform_postorder(comp, _function)
  return counter[0]


def check_has_single_placement(comp, single_placement):
  """Checks that the AST of `comp` contains only `single_placement`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock`.
    single_placement: Instance of `placement_literals.PlacementLiteral` which
      should be the only placement present under `comp`.

  Raises:
    ValueError: If the AST under `comp` contains any
    `computation_types.FederatedType` other than `single_placement`.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(single_placement, placement_literals.PlacementLiteral)

  def _check_single_placement(comp):
    """Checks that the placement in `type_spec` matches `single_placement`."""
    if (isinstance(comp.type_signature, computation_types.FederatedType) and
        comp.type_signature.placement != single_placement):
      raise ValueError('Comp contains a placement other than {}; '
                       'placement {} on comp {} inside the structure. '.format(
                           single_placement, comp.type_signature.placement,
                           comp.compact_representation()))
    return comp, False

  transformation_utils.transform_postorder(comp, _check_single_placement)


def check_intrinsics_whitelisted_for_reduction(comp):
  """Checks whitelist of intrinsics reducible to aggregate or broadcast.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` to check for
      presence of intrinsics not currently immediately reducible to
      `FEDERATED_AGGREGATE` or `FEDERATED_BROADCAST`, or local processing.

  Raises:
    ValueError: If we encounter an intrinsic under `comp` that is not
    whitelisted as currently reducible.
  """
  # TODO(b/135930668): Factor this and other non-transforms (e.g.
  # `check_has_unique_names` out of this file into a structure specified for
  # static analysis of ASTs.
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  uri_whitelist = (
      intrinsic_defs.FEDERATED_AGGREGATE.uri,
      intrinsic_defs.FEDERATED_APPLY.uri,
      intrinsic_defs.FEDERATED_BROADCAST.uri,
      intrinsic_defs.FEDERATED_MAP.uri,
      intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
      intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri,
      intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri,
      intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri,
      intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri,
  )

  def _check_whitelisted(comp):
    if (isinstance(comp, building_blocks.Intrinsic) and
        comp.uri not in uri_whitelist):
      raise ValueError(
          'Encountered an Intrinsic not currently reducible to aggregate or '
          'broadcast, the intrinsic {}'.format(comp.compact_representation()))
    return comp, False

  transformation_utils.transform_postorder(comp, _check_whitelisted)


def check_has_unique_names(comp):
  if not transformation_utils.has_unique_names(comp):
    raise ValueError(
        'This transform should only be called after we have uniquified all '
        '`building_blocks.Reference` names, since we may be moving '
        'computations with unbound references under constructs which bind '
        'those references.')


def extract_nodes_consuming(tree, predicate):
  """Returns the set of AST nodes which consume nodes matching `predicate`.

  Notice we adopt the convention that a node which itself satisfies the
  predicate is in this set.

  Args:
    tree: Instance of `building_blocks.ComputationBuildingBlock` to view as an
      abstract syntax tree, and construct the set of nodes in this tree having a
      dependency on nodes matching `predicate`; that is, the set of nodes whose
      value depends on evaluating nodes matching `predicate`.
    predicate: One-arg callable, accepting arguments of type
      `building_blocks.ComputationBuildingBlock` and returning a `bool`
      indicating match or mismatch with the desired pattern.

  Returns:
    A `set` of `building_blocks.ComputationBuildingBlock` instances
    representing the nodes in `tree` dependent on nodes matching `predicate`.
  """
  py_typecheck.check_type(tree, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_callable(predicate)
  dependent_nodes = set()

  def _are_children_in_dependent_set(comp, symbol_tree):
    """Checks if the dependencies of `comp` are present in `dependent_nodes`."""
    if isinstance(
        comp, (building_blocks.Intrinsic, building_blocks.Data,
               building_blocks.Placement, building_blocks.CompiledComputation)):
      return False
    elif isinstance(comp, building_blocks.Lambda):
      return comp.result in dependent_nodes
    elif isinstance(comp, building_blocks.Block):
      return any(x[1] in dependent_nodes
                 for x in comp.locals) or comp.result in dependent_nodes
    elif isinstance(comp, building_blocks.Tuple):
      return any(x in dependent_nodes for x in comp)
    elif isinstance(comp, building_blocks.Selection):
      return comp.source in dependent_nodes
    elif isinstance(comp, building_blocks.Call):
      return comp.function in dependent_nodes or comp.argument in dependent_nodes
    elif isinstance(comp, building_blocks.Reference):
      return _is_reference_dependent(comp, symbol_tree)

  def _is_reference_dependent(comp, symbol_tree):
    try:
      payload = symbol_tree.get_payload_with_name(comp.name)
    except NameError:
      return False
    # The postorder traversal ensures that we process any
    # bindings before we process the reference to those bindings
    return payload.value in dependent_nodes

  def _populate_dependent_set(comp, symbol_tree):
    """Populates `dependent_nodes` with all nodes dependent on `predicate`."""
    if predicate(comp):
      dependent_nodes.add(comp)
    elif _are_children_in_dependent_set(comp, symbol_tree):
      dependent_nodes.add(comp)
    return comp, False

  symbol_tree = transformation_utils.SymbolTree(
      transformation_utils.ReferenceCounter)
  transformation_utils.transform_postorder_with_symbol_bindings(
      tree, _populate_dependent_set, symbol_tree)
  return dependent_nodes


def check_broadcast_not_dependent_on_aggregate(tree):
  """Raises if any broadcast in `tree` ingests the result of an aggregate.

  We explicitly check for this pattern since if it occurs, `tree` is not
  reducible to broadcast-map-aggregate form.


  Args:
    tree: Instance of `building_blocks.ComputationBuildingBlock` to check for
      the presence of a broadcast which ingests the result of an aggregate.

  Raises:
    ValueError: If a broadcast in `tree` consumes the result of an aggregate.
  """

  py_typecheck.check_type(tree, building_blocks.ComputationBuildingBlock)

  def aggregate_predicate(x):
    return (isinstance(x, building_blocks.Intrinsic) and
            x.uri == intrinsic_defs.FEDERATED_AGGREGATE.uri)

  def broadcast_predicate(x):
    return (isinstance(x, building_blocks.Intrinsic) and
            x.uri == intrinsic_defs.FEDERATED_BROADCAST.uri)

  nodes_dependent_on_aggregate = extract_nodes_consuming(
      tree, aggregate_predicate)

  nodes_dependent_on_broadcast = extract_nodes_consuming(
      tree, broadcast_predicate)

  broadcast_dependent = False
  examples = []

  for node in nodes_dependent_on_aggregate:
    if isinstance(node, building_blocks.Call):
      if (node.argument in nodes_dependent_on_aggregate and
          node.function in nodes_dependent_on_broadcast):
        broadcast_dependent = True
        examples.append(node)
  if broadcast_dependent:
    raise ValueError('Detected broadcast dependent on aggregate. '
                     'Examples are: {}'.format(examples))


def count_tensorflow_ops_under(comp):
  """Counts total TF ops in any TensorFlow computations under `comp`.

  Notice that this function is designed for the purpose of instrumentation,
  in particular to check the size and constituents of the TensorFlow
  artifacts generated.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` whose TF ops we
      wish to count.

  Returns:
    `integer` count of number of TF ops present in any
    `building_blocks.CompiledComputation` of the TensorFlow
    variety under `comp`.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  # TODO(b/129791812): Cleanup Python 2 and 3 compatibility
  total_tf_ops = [0]

  def _count_tf_ops(inner_comp):
    if isinstance(
        inner_comp, building_blocks.CompiledComputation
    ) and inner_comp.proto.WhichOneof('computation') == 'tensorflow':
      total_tf_ops[0] += building_block_analysis.count_tensorflow_ops_in(
          inner_comp)
    return inner_comp, False

  transformation_utils.transform_postorder(comp, _count_tf_ops)
  return total_tf_ops[0]


def count_tensorflow_variables_under(comp):
  """Counts total TF variables in any TensorFlow computations under `comp`.

  Notice that this function is designed for the purpose of instrumentation,
  in particular to check the size and constituents of the TensorFlow
  artifacts generated.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` whose TF
      variables we wish to count.

  Returns:
    `integer` count of number of TF variables present in any
    `building_blocks.CompiledComputation` of the TensorFlow
    variety under `comp`.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  # TODO(b/129791812): Cleanup Python 2 and 3 compatibility
  total_tf_vars = [0]

  def _count_tf_vars(inner_comp):
    if isinstance(
        inner_comp, building_blocks.CompiledComputation
    ) and inner_comp.proto.WhichOneof('computation') == 'tensorflow':
      total_tf_vars[0] += building_block_analysis.count_tensorflow_variables_in(
          inner_comp)
    return inner_comp, False

  transformation_utils.transform_postorder(comp, _count_tf_vars)
  return total_tf_vars[0]
