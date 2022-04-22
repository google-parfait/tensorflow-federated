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
"""A library of transformation functions for ASTs."""

from typing import FrozenSet, Tuple

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import compiled_computation_transforms
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.types import type_transformations

TransformReturnType = Tuple[building_blocks.ComputationBuildingBlock, bool]


def _apply_transforms(comp, transforms):
  """Applies all `transforms` in a single walk of `comp`.

  This function is private for a reason; TFF does not intend to expose the
  capability to chain arbitrary transformations in this way, since the
  application of one transformation may cause the resulting AST to violate the
  assumptions of another. This function should be used quite selectively and
  considered extensively in order to avoid such subtle issues.

  Args:
    comp: An instance of `building_blocks.ComputationBuildingBlock` to transform
      with all elements of `transforms`.
    transforms: An instance of `transformation_utils.TransformSpec` or iterable
      thereof, the transformations to apply to `comp`.

  Returns:
    A transformed version of `comp`, with all transformations in `transforms`
    applied.

  Raises:
    TypeError: If the types don't match.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  if isinstance(transforms, transformation_utils.TransformSpec):
    transforms = [transforms]
  else:
    for transform in transforms:
      py_typecheck.check_type(transform, transformation_utils.TransformSpec)

  def _transform(comp):
    modified = False
    for transform in transforms:
      comp, transform_modified = transform.transform(comp)
      modified = modified or transform_modified
    return comp, modified

  return transformation_utils.transform_postorder(comp, _transform)


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
  return _apply_transforms(comp, RemoveUnusedBlockLocals())


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


def transform_tf_call_ops_to_disable_grappler(comp):
  """Performs grappler disabling on TensorFlow subcomputations."""
  return _apply_transforms(
      comp, compiled_computation_transforms.DisableCallOpGrappler())


def transform_tf_add_ids(comp):
  """Adds unique IDs to each TensorFlow subcomputations."""
  return _apply_transforms(comp, compiled_computation_transforms.AddUniqueIDs())


def check_disallowed_ops(
    comp: building_blocks.ComputationBuildingBlock,
    disallowed_op_names: FrozenSet[str]) -> TransformReturnType:
  """Raises error on disallowed ops in any Tensorflow computation."""
  return _apply_transforms(
      comp,
      compiled_computation_transforms.RaiseOnDisallowedOp(disallowed_op_names))
