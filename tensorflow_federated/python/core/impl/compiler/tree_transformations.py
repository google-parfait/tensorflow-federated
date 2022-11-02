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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A library of transformations for ASTs."""

import collections
from collections.abc import Callable

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_transformations


def remove_mapped_or_applied_identity(comp):
  r"""Removes all the mapped or applied identity functions in `comp`.

  This transform traverses `comp` postorder, matches the following pattern, and
  removes all the mapped or applied identity fucntions by replacing the
  following computation:

            Call
           /    \
  Intrinsic      Tuple
                 |
                 [Lambda(x), Comp(y)]
                           \
                            Ref(x)

  Intrinsic(<(x -> x), y>)

  with its argument:

  Comp(y)

  y

  Args:
    comp: The computation building block in which to perform the removals.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    TypeError: If types do not match.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)

  def _should_transform(comp):
    """Returns `True` if `comp` is a mapped or applied identity function."""
    if (comp.is_call() and comp.function.is_intrinsic() and
        comp.function.uri in (
            intrinsic_defs.FEDERATED_MAP.uri,
            intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
            intrinsic_defs.FEDERATED_APPLY.uri,
            intrinsic_defs.SEQUENCE_MAP.uri,
        )):
      called_function = comp.argument[0]
      return building_block_analysis.is_identity_function(called_function)
    return False

  def _transform(comp):
    if not _should_transform(comp):
      return comp, False
    transformed_comp = comp.argument[1]
    return transformed_comp, True

  return transformation_utils.transform_postorder(comp, _transform)


class RemoveUnusedBlockLocals(transformation_utils.TransformSpec):
  """Removes block local variables which are not used in the result."""

  def should_transform(self, comp):
    return comp.is_block()

  def transform(self, comp):
    if not self.should_transform(comp):
      return comp, False
    unbound_ref_set = transformation_utils.get_map_of_unbound_references(
        comp.result)[comp.result]
    if (not unbound_ref_set) or (not comp.locals):
      return comp.result, True
    new_locals = []
    for name, val in reversed(comp.locals):
      if name in unbound_ref_set:
        new_locals.append((name, val))
        unbound_ref_set = unbound_ref_set.union(
            transformation_utils.get_map_of_unbound_references(val)[val])
        unbound_ref_set.discard(name)
    if len(new_locals) == len(comp.locals):
      return comp, False
    elif not new_locals:
      return comp.result, True
    return building_blocks.Block(reversed(new_locals), comp.result), True


def remove_unused_block_locals(comp):
  transform_spec = RemoveUnusedBlockLocals()
  return transformation_utils.transform_postorder(comp,
                                                  transform_spec.transform)


def uniquify_reference_names(comp, name_generator=None):
  """Replaces all the bound reference names in `comp` with unique names.

  Notice that `uniquify_reference_names` simply leaves alone any reference
  which is unbound under `comp`.

  Args:
    comp: The computation building block in which to perform the replacements.
    name_generator: An optional generator to use for creating unique names. If
      `name_generator` is not None, all existing bindings will be replaced.

  Returns:
    Returns a transformed version of comp inside of which all variable names
    are guaranteed to be unique, and are guaranteed to not mask any unbound
    names referenced in the body of `comp`.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  # Passing `comp` to `unique_name_generator` here will ensure that the
  # generated names conflict with neither bindings in `comp` nor unbound
  # references in `comp`.
  if name_generator is None:
    name_generator = building_block_factory.unique_name_generator(comp)
    rename_all = False
  else:
    # If a `name_generator` was passed in, all bindings must be renamed since
    # we need to avoid duplication with an outer scope.
    rename_all = True
  used_names = set()

  class _RenameNode(transformation_utils.BoundVariableTracker):
    """transformation_utils.SymbolTree node for renaming References in ASTs."""

    def __init__(self, name, value):
      super().__init__(name, value)
      py_typecheck.check_type(name, str)
      if rename_all or name in used_names:
        self.new_name = next(name_generator)
      else:
        self.new_name = name
      used_names.add(self.new_name)

    def __str__(self):
      return 'Value: {}, name: {}, new_name: {}'.format(self.value, self.name,
                                                        self.new_name)

  def _transform(comp, context_tree):
    """Renames References in `comp` to unique names."""
    if comp.is_reference():
      payload = context_tree.get_payload_with_name(comp.name)
      if payload is None:
        return comp, False
      new_name = payload.new_name
      if new_name is comp.name:
        return comp, False
      return building_blocks.Reference(new_name, comp.type_signature,
                                       comp.context), True
    elif comp.is_block():
      new_locals = []
      modified = False
      for name, val in comp.locals:
        context_tree.walk_down_one_variable_binding()
        new_name = context_tree.get_payload_with_name(name).new_name
        modified = modified or (new_name is not name)
        new_locals.append((new_name, val))
      return building_blocks.Block(new_locals, comp.result), modified
    elif comp.is_lambda():
      if comp.parameter_type is None:
        return comp, False
      context_tree.walk_down_one_variable_binding()
      new_name = context_tree.get_payload_with_name(
          comp.parameter_name).new_name
      if new_name is comp.parameter_name:
        return comp, False
      return building_blocks.Lambda(new_name, comp.parameter_type,
                                    comp.result), True
    return comp, False

  symbol_tree = transformation_utils.SymbolTree(_RenameNode)
  return transformation_utils.transform_postorder_with_symbol_bindings(
      comp, _transform, symbol_tree)


def strip_placement(comp):
  """Strips `comp`'s placement, returning a non-federated computation.

  For this function to complete successfully `comp` must:
  1) contain at most one federated placement.
  2) not contain intrinsics besides `apply`, `map`, `zip`, and `federated_value`
  3) not contain `building_blocks.Data` of federated type.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` satisfying the
      assumptions above.

  Returns:
    A modified version of `comp` containing no intrinsics nor any federated
    types or values.

  Raises:
    TypeError: If `comp` is not a building block.
    ValueError: If conditions (1), (2), or (3) above are unsatisfied.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  placement = None
  name_generator = building_block_factory.unique_name_generator(comp)

  def _ensure_single_placement(new_placement):
    nonlocal placement
    if placement is None:
      placement = new_placement
    elif placement != new_placement:
      raise ValueError(
          'Attempted to `strip_placement` from computation containing '
          'multiple different placements.\n'
          f'Found placements `{placement}` and `{new_placement}` in '
          f'comp:\n{comp.compact_representation()}')

  def _remove_placement_from_type(type_spec):
    if type_spec.is_federated():
      _ensure_single_placement(type_spec.placement)
      return type_spec.member, True
    else:
      return type_spec, False

  def _remove_reference_placement(comp):
    """Unwraps placement from references and updates unbound reference info."""
    new_type, _ = type_transformations.transform_type_postorder(
        comp.type_signature, _remove_placement_from_type)
    return building_blocks.Reference(comp.name, new_type)

  def _identity_function(arg_type):
    """Creates `lambda x: x` with argument type `arg_type`."""
    arg_name = next(name_generator)
    val = building_blocks.Reference(arg_name, arg_type)
    lam = building_blocks.Lambda(arg_name, arg_type, val)
    return lam

  def _call_first_with_second_function(fn_type, arg_type):
    """Creates `lambda x: x[0](x[1])` with the provided ."""
    arg_name = next(name_generator)
    tuple_ref = building_blocks.Reference(arg_name, [fn_type, arg_type])
    fn = building_blocks.Selection(tuple_ref, index=0)
    arg = building_blocks.Selection(tuple_ref, index=1)
    called_fn = building_blocks.Call(fn, arg)
    return building_blocks.Lambda(arg_name, tuple_ref.type_signature, called_fn)

  def _call_function(arg_type):
    """Creates `lambda x: x()` argument type `arg_type`."""
    arg_name = next(name_generator)
    arg_ref = building_blocks.Reference(arg_name, arg_type)
    called_arg = building_blocks.Call(arg_ref, None)
    return building_blocks.Lambda(arg_name, arg_type, called_arg)

  def _replace_intrinsics_with_functions(comp):
    """Helper to remove intrinsics from the AST."""
    tys = comp.type_signature

    # These functions have no runtime behavior and only exist to adjust
    # placement. They are replaced here with  `lambda x: x`.
    identities = [
        intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri,
        intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri,
        intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri,
        intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri,
    ]
    if comp.uri in identities:
      return _identity_function(tys.result.member)

    # These functions all `map` a value and are replaced with
    # `lambda args: args[0](args[1])
    maps = [
        intrinsic_defs.FEDERATED_MAP.uri,
        intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
        intrinsic_defs.FEDERATED_APPLY.uri,
    ]
    if comp.uri in maps:
      return _call_first_with_second_function(tys.parameter[0],
                                              tys.parameter[1].member)

    # `federated_eval`'s argument must simply be `call`ed and is replaced
    # with `lambda x: x()`
    evals = [
        intrinsic_defs.FEDERATED_EVAL_AT_SERVER.uri,
        intrinsic_defs.FEDERATED_EVAL_AT_CLIENTS.uri,
    ]
    if comp.uri in evals:
      return _call_function(tys.parameter)

    raise ValueError('Disallowed intrinsic: {}'.format(comp))

  def _remove_lambda_placement(comp):
    """Removes placement from Lambda's parameter."""
    if comp.parameter_name is None:
      new_parameter_type = None
    else:
      new_parameter_type, _ = type_transformations.transform_type_postorder(
          comp.parameter_type, _remove_placement_from_type)
    return building_blocks.Lambda(comp.parameter_name, new_parameter_type,
                                  comp.result)

  def _simplify_calls(comp):
    """Unwraps structures introduced by removing intrinsics."""
    zip_or_value_removed = (
        comp.function.result.is_reference() and
        comp.function.result.name == comp.function.parameter_name)
    if zip_or_value_removed:
      return comp.argument
    else:
      map_removed = (
          comp.function.result.is_call() and
          comp.function.result.function.is_selection() and
          comp.function.result.function.index == 0 and
          comp.function.result.argument.is_selection() and
          comp.function.result.argument.index == 1 and
          comp.function.result.function.source.is_reference() and
          comp.function.result.function.source.name
          == comp.function.parameter_name and
          comp.function.result.function.source.is_reference() and
          comp.function.result.function.source.name
          == comp.function.parameter_name and comp.argument.is_struct())
      if map_removed:
        return building_blocks.Call(comp.argument[0], comp.argument[1])
    return comp

  def _transform(comp):
    """Dispatches to helpers above."""
    if comp.is_reference():
      return _remove_reference_placement(comp), True
    elif comp.is_intrinsic():
      return _replace_intrinsics_with_functions(comp), True
    elif comp.is_lambda():
      return _remove_lambda_placement(comp), True
    elif comp.is_call() and comp.function.is_lambda():
      return _simplify_calls(comp), True
    elif comp.is_data() and comp.type_signature.is_federated():
      raise ValueError(f'Cannot strip placement from federated data: {comp}')
    return comp, False

  return transformation_utils.transform_postorder(comp, _transform)


def _reduce_intrinsic(
    comp, uri, body_fn: Callable[[building_blocks.ComputationBuildingBlock],
                                 building_blocks.ComputationBuildingBlock]):
  """Replaces all the intrinsics with the given `uri` with a callable."""
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(uri, str)

  def _should_transform(comp):
    return comp.is_intrinsic() and comp.uri == uri

  def _transform(comp):
    if not _should_transform(comp):
      return comp, False
    arg_name = next(building_block_factory.unique_name_generator(comp))
    comp_arg = building_blocks.Reference(arg_name,
                                         comp.type_signature.parameter)
    intrinsic_body = body_fn(comp_arg)
    intrinsic_reduced = building_blocks.Lambda(comp_arg.name,
                                               comp_arg.type_signature,
                                               intrinsic_body)
    return intrinsic_reduced, True

  return transformation_utils.transform_postorder(comp, _transform)


def _apply_generic_op(op, arg):
  if not (arg.type_signature.is_federated() or
          type_analysis.is_structure_of_tensors(arg.type_signature)):
    # If there are federated elements nested in a struct, we need to zip these
    # together before passing to binary operator constructor.
    arg = building_block_factory.create_federated_zip(arg)
  return building_block_factory.apply_binary_operator_with_upcast(arg, op)


def get_intrinsic_reductions(
) -> dict[str, Callable[[building_blocks.ComputationBuildingBlock],
                        building_blocks.ComputationBuildingBlock]]:
  """Returns map from intrinsic to reducing function.

  The returned dictionary is a `collections.OrderedDict` which maps intrinsic
  URIs to functions from building-block intrinsic arguments to an implementation
  of the intrinsic call which has been reduced to a smaller, more fundamental
  set of intrinsics.

  Bodies generated by later dictionary entries will not contain references
  to intrinsics whose entries appear earlier in the dictionary. This property
  is useful for simple reduction of an entire computation by iterating through
  the map of intrinsics, substituting calls to each.
  """

  # TODO(b/122728050): Implement reductions that follow roughly the following
  # breakdown in order to minimize the number of intrinsics that backends need
  # to support and maximize opportunities for merging processing logic to keep
  # the number of communication phases as small as it is practical. Perform
  # these reductions before FEDERATED_SUM (more reductions documented below).
  #
  # - FEDERATED_AGGREGATE(x, zero, accu, merge, report) :=
  #     GENERIC_MAP(
  #       GENERIC_REDUCE(
  #         GENERIC_PARTIAL_REDUCE(x, zero, accu, INTERMEDIATE_AGGREGATORS),
  #         zero, merge, SERVER),
  #       report)
  #
  # - FEDERATED_APPLY(f, x) := GENERIC_APPLY(f, x)
  #
  # - FEDERATED_BROADCAST(x) := GENERIC_BROADCAST(x, CLIENTS)
  #
  # - FEDERATED_MAP(f, x) := GENERIC_MAP(f, x)
  #
  # - FEDERATED_VALUE_AT_CLIENTS(x) := GENERIC_PLACE(x, CLIENTS)
  #
  # - FEDERATED_VALUE_AT_SERVER(x) := GENERIC_PLACE(x, SERVER)

  def generic_divide(arg):
    """Divides two arguments when possible."""
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    return _apply_generic_op(tf.divide, arg)

  def generic_multiply(arg):
    """Multiplies two arguments when possible."""
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    return _apply_generic_op(tf.multiply, arg)

  def generic_plus(arg):
    """Adds two arguments when possible."""
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    return _apply_generic_op(tf.add, arg)

  def federated_weighted_mean(arg):
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    w = building_blocks.Selection(arg, index=1)
    multiplied = generic_multiply(arg)
    zip_arg = building_blocks.Struct([(None, multiplied), (None, w)])
    summed = federated_sum(building_block_factory.create_federated_zip(zip_arg))
    return generic_divide(summed)

  def federated_mean(arg):
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    one = building_block_factory.create_generic_constant(arg.type_signature, 1)
    mean_arg = building_blocks.Struct([(None, arg), (None, one)])
    return federated_weighted_mean(mean_arg)

  def federated_sum(x):
    py_typecheck.check_type(x, building_blocks.ComputationBuildingBlock)
    operand_type = x.type_signature.member
    zero = building_block_factory.create_generic_constant(operand_type, 0)
    plus_op = building_block_factory.create_tensorflow_binary_operator_with_upcast(
        tf.add, computation_types.StructType([operand_type, operand_type]))
    identity = building_block_factory.create_identity(operand_type)
    return building_block_factory.create_federated_aggregate(
        x, zero, plus_op, plus_op, identity)

  # - FEDERATED_ZIP(x, y) := GENERIC_ZIP(x, y)
  #
  # - GENERIC_AVERAGE(x: {T}@p, q: placement) :=
  #     GENERIC_WEIGHTED_AVERAGE(x, GENERIC_ONE, q)
  #
  # - GENERIC_WEIGHTED_AVERAGE(x: {T}@p, w: {U}@p, q: placement) :=
  #     GENERIC_MAP(GENERIC_DIVIDE, GENERIC_SUM(
  #       GENERIC_MAP(GENERIC_MULTIPLY, GENERIC_ZIP(x, w)), p))
  #
  #     Note: The above formula does not account for type casting issues that
  #     arise due to the interplay betwen the types of values and weights and
  #     how they relate to types of products and ratios, and either the formula
  #     or the type signatures may need to be tweaked.
  #
  # - GENERIC_SUM(x: {T}@p, q: placement) :=
  #     GENERIC_REDUCE(x, GENERIC_ZERO, GENERIC_PLUS, q)
  #
  # - GENERIC_PARTIAL_SUM(x: {T}@p, q: placement) :=
  #     GENERIC_PARTIAL_REDUCE(x, GENERIC_ZERO, GENERIC_PLUS, q)
  #
  # - GENERIC_AGGREGATE(
  #     x: {T}@p, zero: U, accu: <U,T>->U, merge: <U,U>=>U, report: U->R,
  #     q: placement) :=
  #     GENERIC_MAP(report, GENERIC_REDUCE(x, zero, accu, q))
  #
  # - GENERIC_REDUCE(x: {T}@p, zero: U, op: <U,T>->U, q: placement) :=
  #     GENERIC_MAP((a -> SEQUENCE_REDUCE(a, zero, op)), GENERIC_COLLECT(x, q))
  #
  # - GENERIC_PARTIAL_REDUCE(x: {T}@p, zero: U, op: <U,T>->U, q: placement) :=
  #     GENERIC_MAP(
  #       (a -> SEQUENCE_REDUCE(a, zero, op)), GENERIC_PARTIAL_COLLECT(x, q))
  #
  # - SEQUENCE_SUM(x: T*) :=
  #     SEQUENCE_REDUCE(x, GENERIC_ZERO, GENERIC_PLUS)
  #
  # After performing the full set of reductions, we should only see instances
  # of the following intrinsics in the result, all of which are currently
  # considered non-reducible, and intrinsics such as GENERIC_PLUS should apply
  # only to non-federated, non-sequence types (with the appropriate calls to
  # GENERIC_MAP or SEQUENCE_MAP injected).
  #
  # - GENERIC_APPLY
  # - GENERIC_BROADCAST
  # - GENERIC_COLLECT
  # - GENERIC_DIVIDE
  # - GENERIC_MAP
  # - GENERIC_MULTIPLY
  # - GENERIC_ONE
  # - GENERIC_ONLY
  # - GENERIC_PARTIAL_COLLECT
  # - GENERIC_PLACE
  # - GENERIC_PLUS
  # - GENERIC_ZERO
  # - GENERIC_ZIP
  # - SEQUENCE_MAP
  # - SEQUENCE_REDUCE

  intrinsic_bodies_by_uri = collections.OrderedDict([
      (intrinsic_defs.FEDERATED_MEAN.uri, federated_mean),
      (intrinsic_defs.FEDERATED_WEIGHTED_MEAN.uri, federated_weighted_mean),
      (intrinsic_defs.FEDERATED_SUM.uri, federated_sum),
      (intrinsic_defs.GENERIC_DIVIDE.uri, generic_divide),
      (intrinsic_defs.GENERIC_MULTIPLY.uri, generic_multiply),
      (intrinsic_defs.GENERIC_PLUS.uri, generic_plus),
  ])
  return intrinsic_bodies_by_uri


def replace_intrinsics_with_bodies(comp):
  """Iterates over all intrinsic bodies, inlining the intrinsics in `comp`.

  This function operates on the AST level; meaning, it takes in a
  `building_blocks.ComputationBuildingBlock` as an argument and
  returns one as well. `replace_intrinsics_with_bodies` is intended to be the
  standard reduction function, which will reduce all currently implemented
  intrinsics to their bodies.

  Notice that the success of this function depends on the contract of
  `intrinsic_bodies.get_intrinsic_bodies`, that the dict returned by that
  function is ordered from more complex intrinsic to less complex intrinsics.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` in which we
      wish to replace all intrinsics with their bodies.

  Returns:
    Instance of `building_blocks.ComputationBuildingBlock` with all
    the intrinsics from `intrinsic_bodies.py` inlined with their bodies, along
    with a Boolean indicating whether there was any inlining in fact done.

  Raises:
    TypeError: If the types don't match.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  bodies = get_intrinsic_reductions()
  transformed = False
  for uri, body in bodies.items():
    comp, uri_found = _reduce_intrinsic(comp, uri, body)
    transformed = transformed or uri_found

  return comp, transformed


def _ensure_structure(int_or_structure, int_or_structure_type,
                      possible_struct_type):
  if int_or_structure_type.is_struct() or not possible_struct_type.is_struct():
    return int_or_structure
  else:
    # Broadcast int_or_structure to the same structure as the struct type
    return structure.map_structure(lambda *args: int_or_structure,
                                   possible_struct_type)


def _get_secure_intrinsic_reductions(
) -> dict[str, Callable[[building_blocks.ComputationBuildingBlock],
                        building_blocks.ComputationBuildingBlock]]:
  """Returns map from intrinsic to reducing function.

  The returned dictionary is a `collections.OrderedDict` which maps intrinsic
  URIs to functions from building-block intrinsic arguments to an implementation
  of the intrinsic call which has been reduced to a smaller, more fundamental
  set of intrinsics.

  WARNING: the reductions returned here will produce computaiton bodies that do
  **NOT** perform the crypto protocol. This method is intended only for testing
  settings.

  Bodies generated by later dictionary entries will not contain references
  to intrinsics whose entries appear earlier in the dictionary. This property
  is useful for simple reduction of an entire computation by iterating through
  the map of intrinsics, substituting calls to each.
  """

  def federated_secure_sum(arg):
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    summand_arg = building_blocks.Selection(arg, index=0)
    summand_type = summand_arg.type_signature.member
    max_input_arg = building_blocks.Selection(arg, index=1)
    max_input_type = max_input_arg.type_signature

    # Add the max_value as a second value in the zero, so it can be read during
    # `accumulate` to ensure client summands are valid. The value will be
    # later dropped in `report`.
    #
    # While accumulating summands, we'll assert each summand is less than or
    # equal to max_input. Otherwise the comptuation should issue an error.
    summation_zero = building_block_factory.create_generic_constant(
        summand_type, 0)
    aggregation_zero = building_blocks.Struct([summation_zero, max_input_arg],
                                              container_type=tuple)

    def assert_less_equal_max_and_add(summation_and_max_input, summand):
      summation, original_max_input = summation_and_max_input
      max_input = _ensure_structure(original_max_input, max_input_type,
                                    summand_type)

      # Assert that all coordinates in all tensors are less than the secure sum
      # allowed max input value.
      def assert_all_coordinates_less_equal(x, m):
        return tf.Assert(
            tf.reduce_all(
                tf.less_equal(tf.cast(x, tf.int64), tf.cast(m, tf.int64))), [
                    'client value larger than maximum specified for secure sum',
                    x, 'not less than or equal to', m
                ])

      assert_ops = structure.flatten(
          structure.map_structure(assert_all_coordinates_less_equal, summand,
                                  max_input))
      with tf.control_dependencies(assert_ops):
        return structure.map_structure(tf.add, summation,
                                       summand), original_max_input

    assert_less_equal_and_add = building_block_factory.create_tensorflow_binary_operator(
        assert_less_equal_max_and_add,
        operand_type=aggregation_zero.type_signature,
        second_operand_type=summand_type)

    def nested_plus(a, b):
      return structure.map_structure(tf.add, a, b)

    plus_op = building_block_factory.create_tensorflow_binary_operator(
        nested_plus, operand_type=aggregation_zero.type_signature)

    # In the `report` function we take the summation and drop the second element
    # of the struct (which was holding the max_value).
    drop_max_value_op = building_block_factory.create_tensorflow_unary_operator(
        lambda x: type_conversions.type_to_py_container(x[0], summand_type),
        aggregation_zero.type_signature)

    return building_block_factory.create_federated_aggregate(
        summand_arg, aggregation_zero, assert_less_equal_and_add, plus_op,
        drop_max_value_op)

  def federated_secure_sum_bitwidth(arg):
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    summand_arg = building_blocks.Selection(arg, index=0)
    bitwidth_arg = building_blocks.Selection(arg, index=1)

    # Comptue the max_input value from the provided bitwidth.
    def max_input_from_bitwidth(bitwidth):
      # Secure sum is performed with int64, which has 63 bits, and we need at
      # least one bit to hold the summation of two client values.
      max_secure_sum_bitwidth = 62

      def compute_max_input(bits):
        assert_op = tf.Assert(
            tf.less_equal(bits, max_secure_sum_bitwidth), [
                bits,
                f'is greater than maximum bitwidth {max_secure_sum_bitwidth}'
            ])
        with tf.control_dependencies([assert_op]):
          return tf.math.pow(tf.constant(2, tf.int64), tf.cast(bits,
                                                               tf.int64)) - 1

      return structure.map_structure(compute_max_input, bitwidth)

    compute_max_value_op = building_block_factory.create_tensorflow_unary_operator(
        max_input_from_bitwidth, bitwidth_arg.type_signature)

    max_value = building_blocks.Call(compute_max_value_op, bitwidth_arg)
    return federated_secure_sum(
        building_blocks.Struct([summand_arg, max_value]))

  def federated_secure_modular_sum(arg):
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    arg.type_signature.check_struct()
    if arg.type_signature.is_struct_with_python():
      container_type = arg.type_signature.python_container
    else:
      container_type = None
    summand_arg = building_blocks.Selection(arg, index=0)
    raw_summed_values = building_block_factory.create_federated_sum(summand_arg)

    unplaced_modulus = building_blocks.Selection(arg, index=1)
    placed_modulus = building_block_factory.create_federated_value(
        unplaced_modulus, placements.SERVER)
    modulus_arg = building_block_factory.create_federated_zip(
        building_blocks.Struct([raw_summed_values, placed_modulus],
                               container_type=container_type))

    def map_structure_mod(summed_values, modulus):
      modulus = _ensure_structure(modulus, unplaced_modulus.type_signature,
                                  raw_summed_values.type_signature.member)
      return structure.map_structure(tf.math.mod, summed_values, modulus)

    modulus_fn = building_block_factory.create_tensorflow_binary_operator(
        map_structure_mod,
        operand_type=raw_summed_values.type_signature.member,
        second_operand_type=placed_modulus.type_signature.member)
    modulus_computed = building_block_factory.create_federated_apply(
        modulus_fn, modulus_arg)

    return modulus_computed

  def federated_secure_select(arg):
    py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
    client_keys_arg = building_blocks.Selection(arg, index=0)
    max_key_arg = building_blocks.Selection(arg, index=1)
    server_val_arg = building_blocks.Selection(arg, index=2)
    select_fn_arg = building_blocks.Selection(arg, index=3)
    return building_block_factory.create_federated_select(
        client_keys_arg,
        max_key_arg,
        server_val_arg,
        select_fn_arg,
        secure=False)

  secure_intrinsic_bodies_by_uri = collections.OrderedDict([
      (intrinsic_defs.FEDERATED_SECURE_SUM_BITWIDTH.uri,
       federated_secure_sum_bitwidth),
      (intrinsic_defs.FEDERATED_SECURE_MODULAR_SUM.uri,
       federated_secure_modular_sum),
      (intrinsic_defs.FEDERATED_SECURE_SUM.uri, federated_secure_sum),
      (intrinsic_defs.FEDERATED_SECURE_SELECT.uri, federated_secure_select),
  ])
  return secure_intrinsic_bodies_by_uri


def replace_secure_intrinsics_with_insecure_bodies(comp):
  """Iterates over all secure intrinsic bodies, inlining the intrinsics.

  This function operates on the AST level; meaning, it takes in a
  `building_blocks.ComputationBuildingBlock` as an argument and
  returns one as well. `replace_intrinsics_with_bodies` is intended to be the
  standard reduction function, which will reduce all currently implemented
  intrinsics to their bodies.

  Notice that the success of this function depends on the contract of
  `intrinsic_bodies.get_intrinsic_bodies`, that the dict returned by that
  function is ordered from more complex intrinsic to less complex intrinsics.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` in which we
      wish to replace all intrinsics with their bodies.

  Returns:
    Instance of `building_blocks.ComputationBuildingBlock` with all
    the intrinsics from `intrinsic_bodies.py` inlined with their bodies, along
    with a Boolean indicating whether there was any inlining in fact done.

  Raises:
    TypeError: If the types don't match.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  secure_bodies = _get_secure_intrinsic_reductions()
  transformed = False
  for uri, body in secure_bodies.items():
    comp, uri_found = _reduce_intrinsic(comp, uri, body)
    transformed = transformed or uri_found
  return comp, transformed
