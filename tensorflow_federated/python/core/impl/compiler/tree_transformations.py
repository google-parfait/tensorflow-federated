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
"""A library of transformations for ASTs."""

from collections.abc import Sequence

import federated_language

from tensorflow_federated.python.common_libs import py_typecheck


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
  py_typecheck.check_type(
      comp, federated_language.framework.ComputationBuildingBlock
  )

  def _should_transform(comp):
    """Returns `True` if `comp` is a mapped or applied identity function."""
    if (
        isinstance(comp, federated_language.framework.Call)
        and isinstance(comp.function, federated_language.framework.Intrinsic)
        and comp.function.uri
        in (
            federated_language.framework.FEDERATED_MAP.uri,
            federated_language.framework.FEDERATED_MAP_ALL_EQUAL.uri,
            federated_language.framework.FEDERATED_APPLY.uri,
            federated_language.framework.SEQUENCE_MAP.uri,
        )
    ):
      called_function = comp.argument[0]
      return federated_language.framework.is_identity_function(called_function)
    return False

  def _transform(comp):
    if not _should_transform(comp):
      return comp, False
    transformed_comp = comp.argument[1]
    return transformed_comp, True

  return federated_language.framework.transform_postorder(comp, _transform)


class RemoveUnusedBlockLocals:
  """Removes block local variables which are not used in the result."""

  def should_transform(self, comp):
    return isinstance(comp, federated_language.framework.Block)

  def transform(self, comp):
    """Transforms `comp`."""
    if not self.should_transform(comp):
      return comp, False
    unbound_ref_set = (
        federated_language.framework.get_map_of_unbound_references(comp.result)[
            comp.result
        ]
    )
    if (not unbound_ref_set) or (not comp.locals):
      return comp.result, True
    new_locals = []
    for name, val in reversed(comp.locals):
      if name in unbound_ref_set:
        new_locals.append((name, val))
        unbound_ref_set = unbound_ref_set.union(
            federated_language.framework.get_map_of_unbound_references(val)[val]
        )
        unbound_ref_set.discard(name)
    if len(new_locals) == len(comp.locals):
      return comp, False
    elif not new_locals:
      return comp.result, True
    return (
        federated_language.framework.Block(reversed(new_locals), comp.result),
        True,
    )


def remove_unused_block_locals(comp):
  transform_spec = RemoveUnusedBlockLocals()
  return federated_language.framework.transform_postorder(
      comp, transform_spec.transform
  )


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
  py_typecheck.check_type(
      comp, federated_language.framework.ComputationBuildingBlock
  )
  # Passing `comp` to `unique_name_generator` here will ensure that the
  # generated names conflict with neither bindings in `comp` nor unbound
  # references in `comp`.
  if name_generator is None:
    name_generator = federated_language.framework.unique_name_generator(comp)
    rename_all = False
  else:
    # If a `name_generator` was passed in, all bindings must be renamed since
    # we need to avoid duplication with an outer scope.
    rename_all = True
  used_names = set()

  class _RenameNode(federated_language.framework.BoundVariableTracker):
    """federated_language.framework.SymbolTree node for renaming References in ASTs."""

    def __init__(self, name, value):
      super().__init__(name, value)
      py_typecheck.check_type(name, str)
      if rename_all or name in used_names:
        self.new_name = next(name_generator)
      else:
        self.new_name = name
      used_names.add(self.new_name)

    def __str__(self):
      return 'Value: {}, name: {}, new_name: {}'.format(
          self.value, self.name, self.new_name
      )

  def _transform(comp, context_tree):
    """Renames References in `comp` to unique names."""
    if isinstance(comp, federated_language.framework.Reference):
      payload = context_tree.get_payload_with_name(comp.name)
      if payload is None:
        return comp, False
      new_name = payload.new_name
      if new_name is comp.name:
        return comp, False
      return (
          federated_language.framework.Reference(
              new_name, comp.type_signature, comp.context
          ),
          True,
      )
    elif isinstance(comp, federated_language.framework.Block):
      new_locals = []
      modified = False
      for name, val in comp.locals:
        context_tree.walk_down_one_variable_binding()
        new_name = context_tree.get_payload_with_name(name).new_name
        modified = modified or (new_name is not name)
        new_locals.append((new_name, val))
      return (
          federated_language.framework.Block(new_locals, comp.result),
          modified,
      )
    elif isinstance(comp, federated_language.framework.Lambda):
      if comp.parameter_type is None:
        return comp, False
      context_tree.walk_down_one_variable_binding()
      new_name = context_tree.get_payload_with_name(
          comp.parameter_name
      ).new_name
      if new_name is comp.parameter_name:
        return comp, False
      return (
          federated_language.framework.Lambda(
              new_name, comp.parameter_type, comp.result
          ),
          True,
      )
    return comp, False

  symbol_tree = federated_language.framework.SymbolTree(_RenameNode)
  return federated_language.framework.transform_postorder_with_symbol_bindings(
      comp, _transform, symbol_tree
  )


def normalize_types(comp, normalize_all_equal_bit: bool = True):
  """Normalizes the types under `comp`.

  Any `StructWithPythonType`s are normalized to `StructType`s.

  When `normalize_all_equal_bit` is true, all
  `federated_language.CLIENTS`-placed values will
  have `all_equal False`, and all `federated_language.SERVER`-placed values will
  have
  `all_equal True`. This normalization is needed for MapReduceForm since we rely
  on uniformity of the `all_equal` bit when compiling. For example, the values
  processed on the clients can only be accessed through a `federated_zip`,
  which produces a value with its `all_equal` bit set to `False`. Therefore
  any client processing cannot rely on processing values with `True`
  `all_equal` bits. Notice that `normalize_all_equal_bit` relies on the "normal"
  all_equal bit being inserted in the construction of a new
  `federated_language.FederatedType`;
  the constructor by default sets this bit to match the pattern above, so we
  simply ask it to create a new `federated_language.FederatedType` for us. We
  also replace any
  uses of the FEDERATED_MAP_ALL_EQUAL intrinsic with the FEDERATED_MAP
  intrinsic.

  Args:
    comp: Instance of `federated_language.framework.ComputationBuildingBlock` to
      transform.
    normalize_all_equal_bit: Whether to normalize `all_equal` bits in the placed
      values. Should be set to true when compiling for MapReduceForm and false
      when compiling for DistributeAggregateForm.

  Returns:
    A modified version of `comp` with normalized types.
  """
  py_typecheck.check_type(
      comp, federated_language.framework.ComputationBuildingBlock
  )

  def _normalize_type_signature_helper(type_signature):
    if isinstance(type_signature, federated_language.FederatedType):
      if normalize_all_equal_bit:
        return federated_language.FederatedType(
            type_signature.member, type_signature.placement
        )
    elif isinstance(type_signature, federated_language.StructType):
      new_elements = []
      for element_name, element_type in type_signature.items():
        new_elements.append(
            (element_name, _normalize_type_signature_helper(element_type))
        )
      return federated_language.StructType(new_elements)
    return type_signature

  def _normalize_reference_bit(comp):
    return (
        federated_language.framework.Reference(
            comp.name, _normalize_type_signature_helper(comp.type_signature)
        ),
        True,
    )

  def _normalize_lambda_bit(comp):
    # Note that the lambda result has already been normalized due to the post-
    # order traversal.
    return (
        federated_language.framework.Lambda(
            comp.parameter_name,
            _normalize_type_signature_helper(comp.parameter_type),
            comp.result,
        ),
        True,
    )

  def _normalize_intrinsic_bit(comp):
    """Replaces federated map all equal with federated map."""
    if comp.uri != federated_language.framework.FEDERATED_MAP_ALL_EQUAL.uri:
      return comp, False
    parameter_type = [
        comp.type_signature.parameter[0],
        federated_language.FederatedType(
            comp.type_signature.parameter[1].member, federated_language.CLIENTS
        ),
    ]
    intrinsic_type = federated_language.FunctionType(
        parameter_type,
        federated_language.FederatedType(
            comp.type_signature.result.member, federated_language.CLIENTS
        ),
    )
    new_intrinsic = federated_language.framework.Intrinsic(
        federated_language.framework.FEDERATED_MAP.uri, intrinsic_type
    )
    return new_intrinsic, True

  def _transform_switch(comp):
    if isinstance(comp, federated_language.framework.Reference):
      return _normalize_reference_bit(comp)
    elif isinstance(comp, federated_language.framework.Lambda):
      return _normalize_lambda_bit(comp)
    elif (
        isinstance(comp, federated_language.framework.Intrinsic)
        and normalize_all_equal_bit
    ):
      return _normalize_intrinsic_bit(comp)
    return comp, False

  return federated_language.framework.transform_postorder(
      comp, _transform_switch
  )[0]


def replace_selections(
    bb: federated_language.framework.ComputationBuildingBlock,
    ref_name: str,
    path_to_replacement: dict[
        tuple[int, ...], federated_language.framework.ComputationBuildingBlock
    ],
) -> federated_language.framework.ComputationBuildingBlock:
  """Identifies selection pattern and replaces with new binding.

  Note that this function is somewhat brittle in that it only replaces AST
  fragments of exactly the form `ref_name[i][j][k]` (for path `(i, j, k)`).
  That is, it will not detect `let x = ref_name[i][j] in x[k]` or similar.

  This is only sufficient because, at the point this function has been called,
  called lambdas have been replaced with blocks and blocks have been inlined,
  so there are no reference chains that must be traced back. Any reference which
  would eventually resolve to a part of a lambda's parameter instead refers to
  the parameter directly. Similarly, selections from tuples have been collapsed.
  The remaining concern would be selections via calls to opaque compiled
  computations, which we error on.

  Args:
    bb: Instance of `federated_language.framework.ComputationBuildingBlock` in
      which we wish to replace the selections from reference `ref_name` with any
      path in `paths_to_replacement` with the corresponding building block.
    ref_name: Name of the reference to look for selections from.
    path_to_replacement: A map from selection path to the building block with
      which to replace the selection. Note; it is not valid to specify
      overlapping selection paths (where one path encompasses another).

  Returns:
    A possibly transformed version of `bb` with nodes matching the
    selection patterns replaced.
  """

  def _replace(inner_bb):
    # Start with an empty selection
    path = []
    selection = inner_bb
    while isinstance(selection, federated_language.framework.Selection):
      path.append(selection.as_index())
      selection = selection.source
    # In ASTs like x[0][1], we'll see the last (outermost) selection first.
    path.reverse()
    path = tuple(path)
    if (
        isinstance(selection, federated_language.framework.Reference)
        and selection.name == ref_name
        and path in path_to_replacement
        and path_to_replacement[path].type_signature.is_equivalent_to(
            inner_bb.type_signature
        )
    ):
      return path_to_replacement[path], True
    if (
        isinstance(inner_bb, federated_language.framework.Call)
        and isinstance(
            inner_bb.function, federated_language.framework.CompiledComputation
        )
        and inner_bb.argument is not None
        and isinstance(
            inner_bb.argument, federated_language.framework.Reference
        )
        and inner_bb.argument.name == ref_name
    ):
      raise ValueError(
          'Encountered called graph on reference pattern in TFF '
          'AST; this means relying on pattern-matching when '
          'rebinding arguments may be insufficient. Ensure that '
          'arguments are rebound before decorating references '
          'with called identity graphs.'
      )
    return inner_bb, False

  # TODO: b/266705611 - Consider switching to preorder traversal to provide more
  # protection against triggering multiple replacements for nested selections
  # (the type signature check above does provide one layer of protection
  # already).
  result, _ = federated_language.framework.transform_postorder(bb, _replace)
  return result


class ParameterSelectionError(TypeError):

  def __init__(self, path, bb):
    message = (
        'Attempted to rebind references to parameter selection path '
        f'{path}, which is not a valid selection from type '
        f'{bb.parameter_type}. Original AST:\n{bb}'
    )
    super().__init__(message)


def as_function_of_some_subparameters(
    bb: federated_language.framework.Lambda,
    paths: Sequence[Sequence[int]],
) -> federated_language.framework.Lambda:
  """Turns `x -> ...only uses parts of x...` into `parts_of_x -> ...`.

  The names of locals in blocks are not modified, but unused block locals
  are removed.

  If the body of the computation requires more than the allowed subparameters,
  the returned computation will have unbound references.

  Args:
    bb: Instance of `federated_language.framework.Lambda` that we wish to
      rewrite as a function of some subparameters.
    paths: List of the paths representing the input subparameters to use. Each
      path is a tuple of ints (e.g. (5, 3) would represent a selection into the
      original arg like arg[5][3]). Note; it is not valid to specify overlapping
      selection paths (where one path encompasses another).

  Returns:
    An instance of `federated_language.framework.Lambda` with a struct input
    parameter where
    the ith element in the input parameter corresponds to the ith provided path.

  Raises:
    ParameterSelectionError: If a requested path is not possible for the
      original input parameter type.
  """

  def _get_block_local_names(comp):
    names = []

    def _visit(comp):
      if isinstance(comp, federated_language.framework.Block):
        for name, _ in comp.locals:
          names.append(name)

    federated_language.framework.visit_postorder(comp, _visit)
    return names

  federated_language.framework.check_has_unique_names(bb)
  original_local_names = _get_block_local_names(bb)
  bb, _ = remove_unused_block_locals(bb)

  name_generator = federated_language.framework.unique_name_generator(bb)

  type_list = []
  int_paths = []
  for path in paths:
    selected_type = bb.parameter_type
    int_path = []
    for index in path:
      if not isinstance(selected_type, federated_language.StructType):
        raise ParameterSelectionError(path, bb)
      py_typecheck.check_type(index, int)
      if index >= len(selected_type):
        raise ParameterSelectionError(path, bb)
      int_path.append(index)
      selected_type = selected_type[index]
    int_paths.append(tuple(int_path))
    type_list.append(selected_type)

  ref_to_struct = federated_language.framework.Reference(
      next(name_generator), federated_language.StructType(type_list)
  )
  path_to_replacement = {}
  for i, path in enumerate(int_paths):
    path_to_replacement[path] = federated_language.framework.Selection(
        ref_to_struct, index=i
    )

  new_lambda_body = replace_selections(
      bb.result, bb.parameter_name, path_to_replacement
  )
  # Normalize the body so that it is a block.
  if not isinstance(new_lambda_body, federated_language.framework.Block):
    new_lambda_body = federated_language.framework.Block([], new_lambda_body)
  lambda_with_zipped_param = federated_language.framework.Lambda(
      ref_to_struct.name, ref_to_struct.type_signature, new_lambda_body
  )

  new_local_names = _get_block_local_names(lambda_with_zipped_param)
  assert set(new_local_names).issubset(set(original_local_names))

  return lambda_with_zipped_param


def strip_placement(comp):
  """Strips `comp`'s placement, returning a non-federated computation.

  For this function to complete successfully `comp` must:
  1) contain at most one federated placement.
  2) not contain intrinsics besides `apply`, `map`, `zip`, and `federated_value`
  3) not contain `federated_language.framework.Data` of federated type.

  Args:
    comp: Instance of `federated_language.framework.ComputationBuildingBlock`
      satisfying the assumptions above.

  Returns:
    A modified version of `comp` containing no intrinsics nor any federated
    types or values.

  Raises:
    TypeError: If `comp` is not a building block.
    ValueError: If conditions (1), (2), or (3) above are unsatisfied.
  """
  py_typecheck.check_type(
      comp, federated_language.framework.ComputationBuildingBlock
  )
  placement = None
  name_generator = federated_language.framework.unique_name_generator(comp)

  def _ensure_single_placement(new_placement):
    nonlocal placement
    if placement is None:
      placement = new_placement
    elif placement != new_placement:
      raise ValueError(
          'Attempted to `strip_placement` from computation containing '
          'multiple different placements.\n'
          f'Found placements `{placement}` and `{new_placement}` in '
          f'comp:\n{comp.compact_representation()}'
      )

  def _remove_placement_from_type(type_spec):
    if isinstance(type_spec, federated_language.FederatedType):
      _ensure_single_placement(type_spec.placement)
      return type_spec.member, True
    else:
      return type_spec, False

  def _remove_reference_placement(comp):
    """Unwraps placement from references and updates unbound reference info."""
    new_type, _ = federated_language.framework.transform_type_postorder(
        comp.type_signature, _remove_placement_from_type
    )
    return federated_language.framework.Reference(comp.name, new_type)

  def _identity_function(arg_type):
    """Creates `lambda x: x` with argument type `arg_type`."""
    arg_name = next(name_generator)
    val = federated_language.framework.Reference(arg_name, arg_type)
    lam = federated_language.framework.Lambda(arg_name, arg_type, val)
    return lam

  def _call_first_with_second_function(fn_type, arg_type):
    """Creates `lambda x: x[0](x[1])` with the provided ."""
    arg_name = next(name_generator)
    tuple_ref = federated_language.framework.Reference(
        arg_name, [fn_type, arg_type]
    )
    fn = federated_language.framework.Selection(tuple_ref, index=0)
    arg = federated_language.framework.Selection(tuple_ref, index=1)
    called_fn = federated_language.framework.Call(fn, arg)
    return federated_language.framework.Lambda(
        arg_name, tuple_ref.type_signature, called_fn
    )

  def _call_function(arg_type):
    """Creates `lambda x: x()` argument type `arg_type`."""
    arg_name = next(name_generator)
    arg_ref = federated_language.framework.Reference(arg_name, arg_type)
    called_arg = federated_language.framework.Call(arg_ref, None)
    return federated_language.framework.Lambda(arg_name, arg_type, called_arg)

  def _replace_intrinsics_with_functions(comp):
    """Helper to remove intrinsics from the AST."""
    tys = comp.type_signature

    # These functions have no runtime behavior and only exist to adjust
    # placement. They are replaced here with  `lambda x: x`.
    identities = [
        federated_language.framework.FEDERATED_ZIP_AT_SERVER.uri,
        federated_language.framework.FEDERATED_ZIP_AT_CLIENTS.uri,
        federated_language.framework.FEDERATED_VALUE_AT_SERVER.uri,
        federated_language.framework.FEDERATED_VALUE_AT_CLIENTS.uri,
    ]
    if comp.uri in identities:
      return _identity_function(tys.result.member)

    # These functions all `map` a value and are replaced with
    # `lambda args: args[0](args[1])
    maps = [
        federated_language.framework.FEDERATED_MAP.uri,
        federated_language.framework.FEDERATED_MAP_ALL_EQUAL.uri,
        federated_language.framework.FEDERATED_APPLY.uri,
    ]
    if comp.uri in maps:
      return _call_first_with_second_function(
          tys.parameter[0], tys.parameter[1].member
      )

    # `federated_eval`'s argument must simply be `call`ed and is replaced
    # with `lambda x: x()`
    evals = [
        federated_language.framework.FEDERATED_EVAL_AT_SERVER.uri,
        federated_language.framework.FEDERATED_EVAL_AT_CLIENTS.uri,
    ]
    if comp.uri in evals:
      return _call_function(tys.parameter)

    raise ValueError('Disallowed intrinsic: {}'.format(comp))

  def _remove_lambda_placement(comp):
    """Removes placement from Lambda's parameter."""
    if comp.parameter_name is None:
      new_parameter_type = None
    else:
      new_parameter_type, _ = (
          federated_language.framework.transform_type_postorder(
              comp.parameter_type, _remove_placement_from_type
          )
      )
    return federated_language.framework.Lambda(
        comp.parameter_name, new_parameter_type, comp.result
    )

  def _simplify_calls(comp):
    """Unwraps structures introduced by removing intrinsics."""
    zip_or_value_removed = (
        isinstance(comp.function.result, federated_language.framework.Reference)
        and comp.function.result.name == comp.function.parameter_name
    )
    if zip_or_value_removed:
      return comp.argument
    else:
      map_removed = (
          isinstance(comp.function.result, federated_language.framework.Call)
          and isinstance(
              comp.function.result.function,
              federated_language.framework.Selection,
          )
          and comp.function.result.function.index == 0
          and isinstance(
              comp.function.result.argument,
              federated_language.framework.Selection,
          )
          and comp.function.result.argument.index == 1
          and isinstance(
              comp.function.result.function.source,
              federated_language.framework.Reference,
          )
          and comp.function.result.function.source.name
          == comp.function.parameter_name
          and isinstance(
              comp.function.result.function.source,
              federated_language.framework.Reference,
          )
          and comp.function.result.function.source.name
          == comp.function.parameter_name
          and isinstance(comp.argument, federated_language.framework.Struct)
      )
      if map_removed:
        return federated_language.framework.Call(
            comp.argument[0], comp.argument[1]
        )
    return comp

  def _transform(comp):
    """Dispatches to helpers above."""
    if isinstance(comp, federated_language.framework.Reference):
      return _remove_reference_placement(comp), True
    elif isinstance(comp, federated_language.framework.Intrinsic):
      return _replace_intrinsics_with_functions(comp), True
    elif isinstance(comp, federated_language.framework.Lambda):
      return _remove_lambda_placement(comp), True
    elif isinstance(comp, federated_language.framework.Call) and isinstance(
        comp.function, federated_language.framework.Lambda
    ):
      return _simplify_calls(comp), True
    elif isinstance(comp, federated_language.framework.Data) and isinstance(
        comp.type_signature, federated_language.FederatedType
    ):
      raise ValueError(f'Cannot strip placement from federated data: {comp}')
    return comp, False

  return federated_language.framework.transform_postorder(comp, _transform)
