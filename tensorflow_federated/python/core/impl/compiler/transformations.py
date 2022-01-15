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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A library of composite transformation functions.

A composite transformation is one that applies multiple atomic transformation to
an AST either pointwise or serially.
"""

import collections
from typing import Dict, List, Mapping, Set, Tuple

import attr

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import compiled_computation_transforms
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_to_cc_transformations
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.compiler.tree_transformations import extract_computations
from tensorflow_federated.python.core.impl.compiler.tree_transformations import remove_duplicate_block_locals
from tensorflow_federated.python.core.impl.compiler.tree_transformations import remove_mapped_or_applied_identity
from tensorflow_federated.python.core.impl.compiler.tree_transformations import replace_called_lambda_with_block
from tensorflow_federated.python.core.impl.compiler.tree_transformations import uniquify_reference_names
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis


def remove_duplicate_building_blocks(comp):
  """Composite transformation to remove duplicated building blocks."""
  mutated = False
  for transform in [
      replace_called_lambda_with_block,
      remove_mapped_or_applied_identity,
      uniquify_reference_names,
      extract_computations,
      remove_duplicate_block_locals,
  ]:
    comp, comp_mutated = transform(comp)
    mutated = mutated or comp_mutated
  return comp, mutated


def prepare_for_rebinding(comp):
  """Prepares `comp` for extracting rebound variables.

  Currently, this means replacing all called lambdas and inlining all blocks.
  This does not necessarly guarantee that the resulting computation has no
  called lambdas, it merely reduces a level of indirection here. This reduction
  has proved sufficient for identifying variables which are about to be rebound
  in the top-level lambda, necessarily when compiler components factor work out
  from a single function into multiple functions. Since this function makes no
  guarantees about sufficiency, it is the responsibility of the caller to
  ensure that no unbound variables are introduced during the rebinding.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` from which all
      occurrences of a given variable need to be extracted and rebound.

  Returns:
    Another instance of `building_blocks.ComputationBuildingBlock` which has
    had all called lambdas replaced by blocks, all blocks inlined and all
    selections from tuples collapsed.
  """
  # TODO(b/146430051): Follow up here and consider removing or enforcing more
  # strict output invariants when `remove_called_lambdas_and_blocks` is moved
  # in here.
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  comp, _ = tree_transformations.uniquify_reference_names(comp)
  comp, _ = tree_transformations.replace_called_lambda_with_block(comp)
  block_inliner = tree_transformations.InlineBlock(comp)
  selection_replacer = tree_transformations.ReplaceSelectionFromTuple()
  transforms = [block_inliner, selection_replacer]
  symbol_tree = transformation_utils.SymbolTree(
      transformation_utils.ReferenceCounter)

  def _transform_fn(comp, symbol_tree):
    """Transform function chaining inlining and collapsing selections."""
    modified = False
    for transform in transforms:
      if transform.global_transform:
        comp, transform_modified = transform.transform(comp, symbol_tree)
      else:
        comp, transform_modified = transform.transform(comp)
      modified = modified or transform_modified
    return comp, modified

  return transformation_utils.transform_postorder_with_symbol_bindings(
      comp, _transform_fn, symbol_tree)


def _type_contains_function(t):
  return type_analysis.contains(t, lambda x: x.is_function())


def _disallow_higher_order(comp, global_comp):
  if comp.type_signature.is_function():
    param_type = comp.type_signature.parameter
    if param_type is not None and _type_contains_function(param_type):
      raise ValueError(
          f'{comp}\nin\n{global_comp}\nhas higher-order parameter of type {param_type}'
      )
    result_type = comp.type_signature.result
    if _type_contains_function(result_type):
      raise ValueError(
          f'{comp}\nin\n{global_comp}\nhas a higher-order result of type {result_type}'
      )


def transform_to_local_call_dominant(
    comp: building_blocks.Lambda) -> building_blocks.Lambda:
  """Transforms local (non-federated) computations into call-dominant form.

  Args:
    comp: A local computation. Local computations must not contain intrinsics
      or compiled computations with higher-order parameters or results, such as
      the `federated_x` set of intrinsics.

  Returns:
    A transformed but semantically-equivalent `comp`. The resulting `comp` will
    be in CDF (call-dominant form), as defined by the following CFG:

    External -> Intrinsic | Data | Compiled Computation

    CDFElem ->
       External
     | Reference to a bound call to an External
     | Selection(CDFElem, index)
     | Lambda(Block([bindings for External calls, CDF))

    CDF ->
       CDFElem
     | Struct(CDF, ...)
     | Lambda(CDF)
     | Lambda(Block([bindings for External calls], CDF))
  """
  # Top-level comp must be a lambda to ensure that we create a set of bindings
  # immediately under it, as `_build` does for all lambdas.
  global_comp = comp
  name_generator = building_block_factory.unique_name_generator(comp)

  class _Scope():
    """Name resolution scopes which track the creation of new value bindings."""

    def __init__(self, parent=None, bind_to_parent=False):
      """Create a new scope.

      Args:
        parent: An optional parent scope.
        bind_to_parent: If true, `create_bindings` calls will be propagated to
          the parent scope, causing newly-created bindings to be visible at a
          higher level. If false, `create_bindings` will create a new binding in
          this scope. New bindings will be used as locals inside of
          `bindings_to_block_with_result`.
      """
      if parent is None and bind_to_parent:
        raise ValueError('Cannot `bind_to_parent` for `None` parent.')
      self._parent = parent
      self._newly_bound_values = None if bind_to_parent else []
      self._locals = {}

    def resolve(self, name: str):
      if name in self._locals:
        return self._locals[name]
      if self._parent is None:
        raise ValueError(f'Unable to resolve name `{name}` in `{global_comp}`')
      return self._parent.resolve(name)

    def add_local(self, name, value):
      self._locals[name] = value

    def create_binding(self, value):
      """Add a binding to the nearest binding scope."""
      if self._newly_bound_values is None:
        return self._parent.create_binding(value)
      else:
        name = next(name_generator)
        self._newly_bound_values.append((name, value))
        reference = building_blocks.Reference(name, value.type_signature)
        self._locals[name] = reference
        return reference

    def new_child(self):
      return _Scope(parent=self, bind_to_parent=True)

    def new_child_with_bindings(self):
      """Creates a child scope which will hold its own bindings."""
      # NOTE: should always be paired with a call to
      # `bindings_to_block_with_result`.
      return _Scope(parent=self, bind_to_parent=False)

    def bindings_to_block_with_result(self, result):
      # Don't create unnecessary blocks if there aren't any locals.
      if len(self._newly_bound_values) == 0:  # pylint: disable=g-explicit-length-test
        return result
      else:
        return building_blocks.Block(self._newly_bound_values, result)

  def _build(comp, scope):
    """Transforms `comp` to CDF, possibly adding bindings to `scope`."""
    # The structure returned by this function is a generalized version of
    # call-dominant form. This function may result in the patterns specified in
    # the top-level function's docstring.
    if comp.is_reference():
      return scope.resolve(comp.name)
    elif comp.is_selection():
      source = _build(comp.source, scope)
      if source.is_struct():
        return source[comp.as_index()]
      return building_blocks.Selection(source, index=comp.as_index())
    elif comp.is_struct():
      elements = []
      for (name, value) in structure.iter_elements(comp):
        value = _build(value, scope)
        elements.append((name, value))
      return building_blocks.Struct(elements)
    elif comp.is_call():
      function = _build(comp.function, scope)
      argument = None if comp.argument is None else _build(comp.argument, scope)
      if function.is_lambda():
        if argument is not None:
          scope = scope.new_child()
          scope.add_local(function.parameter_name, argument)
        return _build(function.result, scope)
      else:
        return scope.create_binding(building_blocks.Call(function, argument))
    elif comp.is_lambda():
      scope = scope.new_child_with_bindings()
      if comp.parameter_name:
        scope.add_local(
            comp.parameter_name,
            building_blocks.Reference(comp.parameter_name, comp.parameter_type))
      result = _build(comp.result, scope)
      block = scope.bindings_to_block_with_result(result)
      return building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                    block)
    elif comp.is_block():
      scope = scope.new_child()
      for (name, value) in comp.locals:
        scope.add_local(name, _build(value, scope))
      return _build(comp.result, scope)
    elif (comp.is_intrinsic() or comp.is_data() or
          comp.is_compiled_computation()):
      _disallow_higher_order(comp, global_comp)
      return comp
    elif comp.is_placement():
      raise ValueError(f'Found placement {comp} in\n{global_comp}\n'
                       'but placements are not allowed in local computations.')
    else:
      raise ValueError(
          f'Unrecognized computation kind\n{comp}\nin\n{global_comp}')

  scope = _Scope()
  result = _build(comp, scope)
  return scope.bindings_to_block_with_result(result)


def _generate_simple_tensorflow(comp):
  """Naively generates TensorFlow to represent `comp`."""
  tf_parser_callable = tree_to_cc_transformations.TFParser()
  comp, _ = tree_transformations.insert_called_tf_identity_at_leaves(comp)
  comp, _ = transformation_utils.transform_postorder(comp, tf_parser_callable)
  return comp


def construct_tensorflow_calling_lambda_on_concrete_arg(
    parameter: building_blocks.Reference,
    body: building_blocks.ComputationBuildingBlock,
    concrete_arg: building_blocks.ComputationBuildingBlock):
  """Generates TensorFlow for lambda invocation with given arg, body and param.

  That is, generates TensorFlow block encapsulating the logic represented by
  invoking a function with parameter `parameter` and body `body`, with argument
  `concrete_arg`.

  Via the guarantee made in `compiled_computation_transforms.StructCalledGraphs`
  this function makes the claim that the computations which define
  `concrete_arg` will be executed exactly once in the generated TenosorFlow.

  Args:
    parameter: Instance of `building_blocks.Reference` defining the parameter of
      the function to be generated and invoked, as described above. After
      calling this transformation, every instance of  parameter` in `body` will
      represent a reference to `concrete_arg`.
    body: `building_blocks.ComputationBuildingBlock` representing the body of
      the function for which we are generating TensorFlow.
    concrete_arg: `building_blocks.ComputationBuildingBlock` representing the
      argument to be passed to the resulting function. `concrete_arg` will then
      be referred to by every occurrence of `parameter` in `body`. Therefore
      `concrete_arg` must have an equivalent type signature to that of
      `parameter`.

  Returns:
    A called `building_blocks.CompiledComputation`, as specified above.

  Raises:
    TypeError: If the arguments are of the wrong types, or the type signature
      of `concrete_arg` does not match that of `parameter`.
  """
  py_typecheck.check_type(parameter, building_blocks.Reference)
  py_typecheck.check_type(body, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(concrete_arg,
                          building_blocks.ComputationBuildingBlock)
  parameter.type_signature.check_equivalent_to(concrete_arg.type_signature)

  encapsulating_lambda = _generate_simple_tensorflow(
      building_blocks.Lambda(parameter.name, parameter.type_signature, body))
  comp_called = _generate_simple_tensorflow(
      building_blocks.Call(encapsulating_lambda, concrete_arg))
  return comp_called


def _replace_references_in_comp_with_selections_from_arg(
    comp: building_blocks.ComputationBuildingBlock,
    arg_ref: building_blocks.Reference, name_to_output_index: Mapping[str,
                                                                      int]):
  """Uses `name_to_output_index` to rebind references in `comp`."""

  def _replace_values_with_selections(inner_comp):
    if inner_comp.is_reference():
      selected_index = name_to_output_index[inner_comp.name]
      return building_blocks.Selection(
          source=arg_ref, index=selected_index), True
    return inner_comp, False

  comp_replaced, _ = transformation_utils.transform_postorder(
      comp, _replace_values_with_selections)
  return comp_replaced


def _construct_tensorflow_representing_single_local_assignment(
    arg_ref, arg_class, previous_output, name_to_output_index):
  """Constructs TensorFlow to represent assignment to a block local in sequence.

  Creates a tuple which represents all computations in the block local sequence
  depending on those variables which have already been processed, by combining
  the elements of `previous_output` with the computations in `arg_class`. Then
  generates TensorFlow to capture the logic this tuple encapsulates.

  Args:
    arg_ref: `building_blocks.Reference` to use in representing
      `previous_output` inside the body of the Lambda to be parsed to
      TensorFlow. Notice that this is here for name safety.
    arg_class: `list` of `building_blocks.ComputationBuildingBlock`s which are
      dependent on the block local being processed or any preceding block local;
      this should be one of the classes resulting from
      `group_block_locals_by_namespace`.
    previous_output: The result of parsing previous block local bindings into
      functions in the same manner.
    name_to_output_index: `dict` mapping block local variables to their index in
      the result of the generated TensorFlow. This is used to resolve references
      in the computations of `arg_class`, but will not be modified.

  Returns:
    Called instance of `building_blocks.CompiledComputation` representing
    the tuple described above.
  """
  pass_through_args = [
      building_blocks.Selection(source=arg_ref, index=idx)
      for idx, _ in enumerate(previous_output.type_signature)
  ]

  vals_replaced = [
      _replace_references_in_comp_with_selections_from_arg(
          c, arg_ref, name_to_output_index) for c in arg_class
  ]
  return_tuple = building_blocks.Struct(pass_through_args + vals_replaced)

  comp_called = construct_tensorflow_calling_lambda_on_concrete_arg(
      arg_ref, return_tuple, previous_output)
  return comp_called


def _get_unbound_ref(block):
  """Helper to get unbound ref name and type spec if it exists in `block`."""
  all_unbound_refs = transformation_utils.get_map_of_unbound_references(block)
  top_level_unbound_ref = all_unbound_refs[block]
  num_unbound_refs = len(top_level_unbound_ref)
  if num_unbound_refs == 0:
    return None
  elif num_unbound_refs > 1:
    raise ValueError('`create_tensorflow_representing_block` must be passed '
                     'a block with at most a single unbound reference; '
                     'encountered the block {} with {} unbound '
                     'references.'.format(block, len(top_level_unbound_ref)))

  unbound_ref_name = top_level_unbound_ref.pop()

  top_level_type_spec = None

  def _get_unbound_ref_type_spec(inner_comp):
    if (inner_comp.is_reference() and inner_comp.name == unbound_ref_name):
      nonlocal top_level_type_spec
      top_level_type_spec = inner_comp.type_signature
    return inner_comp, False

  transformation_utils.transform_postorder(block, _get_unbound_ref_type_spec)
  return building_blocks.Reference(unbound_ref_name, top_level_type_spec)


def _check_parameters_for_tf_block_generation(block):
  """Helper to validate parameters for parsing block locals into TF graphs."""
  py_typecheck.check_type(block, building_blocks.Block)
  for _, comp in block.locals:
    if not (comp.is_call() and comp.function.is_compiled_computation()):
      raise ValueError(
          'create_tensorflow_representing_block may only be called '
          'on a block whose local variables are all bound to '
          'called TensorFlow computations; encountered a local '
          'bound to {}'.format(comp))

  def _check_contains_only_refs_sels_and_tuples(inner_comp):
    if not (inner_comp.is_reference() or inner_comp.is_selection() or
            inner_comp.is_struct()):
      raise ValueError(
          'create_tensorflow_representing_block may only be called '
          'on a block whose result contains only Selections, '
          'Tuples and References; encountered the building block '
          '{}.'.format(inner_comp))
    return inner_comp, False

  transformation_utils.transform_postorder(
      block.result, _check_contains_only_refs_sels_and_tuples)


def create_tensorflow_representing_block(block):
  """Generates non-duplicated TensorFlow for Block locals binding called graphs.

  Assuming that the argument `block` satisfies the following conditions:

  1. The local variables in `block` are all called graphs, with arbitrary
      arguments.
  2. The result of the Block contains tuples, selections and references,
     but nothing else.

  Then `create_tensorflow_representing_block` will generate a structure, which
  may contain tensorflow functions, calls to tensorflow functions, and
  references, but which have generated this TensorFlow code without duplicating
  work done by referencing the block locals.

  Args:
    block: Instance of `building_blocks.Block`, whose local variables are all
      called instances of `building_blocks.CompiledComputation`, and whose
      result contains only instances of `building_blocks.Reference`,
      `building_blocks.Selection` or `building_blocks.Struct`.

  Returns:
    A transformed version of `block`, which has pushed references to the called
    graphs in the locals of `block` into TensorFlow.

  Raises:
    TypeError: If `block` is not an instance of `building_blocks.Block`.
    ValueError: If the locals of `block` are anything other than called graphs,
      or if the result of `block` contains anything other than selections,
      references and tuples.
  """
  _check_parameters_for_tf_block_generation(block)

  name_generator = building_block_factory.unique_name_generator(block)

  def _construct_reference_representing(comp_to_represent):
    """Helper closing over `name_generator` for name safety."""
    arg_type = comp_to_represent.type_signature
    arg_name = next(name_generator)
    return building_blocks.Reference(arg_name, arg_type)

  top_level_ref = _get_unbound_ref(block)
  named_comp_classes = tree_transformations.group_block_locals_by_namespace(
      block)

  if top_level_ref:
    first_comps = [x[1] for x in named_comp_classes[0]]
    tup = building_blocks.Struct([top_level_ref] + first_comps)
    graph_tup = _generate_simple_tensorflow(tup)
    output_comp = construct_tensorflow_calling_lambda_on_concrete_arg(
        top_level_ref, graph_tup, top_level_ref)
    name_to_output_index = {top_level_ref.name: 0}
  else:
    output_comp = building_block_factory.create_compiled_empty_tuple()
    name_to_output_index = {}

  block_local_names = [x[0] for x in block.locals]

  def _update_name_to_output_index(name_class):
    """Helper closing over `name_to_output_index` and `block_local_names`."""
    offset = len(name_to_output_index.keys())
    for idx, comp_name in enumerate(name_class):
      for var_name in block_local_names:
        if var_name == comp_name:
          name_to_output_index[var_name] = idx + offset

  if top_level_ref:
    first_names = [x[0] for x in named_comp_classes[0]]
    _update_name_to_output_index(first_names)
    remaining_comp_classes = named_comp_classes[1:]
  else:
    remaining_comp_classes = named_comp_classes[:]

  for named_comp_class in remaining_comp_classes:
    if named_comp_class:
      comp_class = [x[1] for x in named_comp_class]
      name_class = [x[0] for x in named_comp_class]
      arg_ref = _construct_reference_representing(output_comp)
      output_comp = _construct_tensorflow_representing_single_local_assignment(
          arg_ref, comp_class, output_comp, name_to_output_index)
      _update_name_to_output_index(name_class)

  arg_ref = _construct_reference_representing(output_comp)
  result_replaced = _replace_references_in_comp_with_selections_from_arg(
      block.result, arg_ref, name_to_output_index)
  comp_called = construct_tensorflow_calling_lambda_on_concrete_arg(
      arg_ref, result_replaced, output_comp)

  return comp_called, True


def generate_tensorflow_for_local_computation(comp):
  """Generates TensorFlow for a local TFF computation.

  This function performs a deduplication of function invocations
  according to `tree_analysis.trees_equal`, and hence may reduce the number
  of calls under `comp`.

  We assume `comp` has type which can be represented by either a call to a
  no-arg `building_blocks.CompiledComputation` of type `tensorflow`, or such a
  `building_blocks.CompiledComputation` itself. That is, the type signature of
  `comp` must be either a potentially nested structure of
  `computation_types.TensorType`s and `computation_types.SequenceType`s, or a
  function whose parameter and return types are such potentially nested
  structures.

  Further, we assume that there are no intrinsic or data building blocks inside
  `comp`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` for which we
      wish to generate TensorFlow.

  Returns:
    Either a called instance of `building_blocks.CompiledComputation` or a
    `building_blocks.CompiledComputation` itself, depending on whether `comp`
    is of non-functional or functional type respectively. Additionally, returns
    a boolean to match the `transformation_utils.TransformSpec` pattern.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  names_uniquified, _ = tree_transformations.uniquify_reference_names(comp)
  # We ensure the argument to `transform_to_local_call_dominant` is a Lambda, as
  # required.
  lambda_wrapping_comp = building_blocks.Lambda(None, None, names_uniquified)
  # CFG for local CDF plus the type of `lambda_wrapping_comp` imply result must
  # be another no-arg lambda.
  local_cdf_comp = transform_to_local_call_dominant(lambda_wrapping_comp).result

  def _package_as_deduplicated_block(inner_comp):
    repacked_block, _ = tree_transformations.remove_duplicate_block_locals(
        inner_comp)
    if not repacked_block.is_block():
      repacked_block = building_blocks.Block([], repacked_block)
    return repacked_block

  if local_cdf_comp.type_signature.is_function():
    # The CFG for local call dominant tells us that the following patterns are
    # possible for a functional computation respecting the structural
    # restrictions we require for `comp`:
    #   1. CompiledComputation
    #   2. Block(bindings, CompiledComp)
    #   3. Block(bindings, Lambda(non-functional result with at most one Block))
    #   4. Lambda(non-functional result with at most one Block)
    if local_cdf_comp.is_compiled_computation():
      # Case 1.
      return local_cdf_comp, not comp.is_compiled_computation()
    elif local_cdf_comp.is_block():
      if local_cdf_comp.result.is_compiled_computation():
        # Case 2. The bindings in `comp` cannot be referenced in `comp.result`;
        # we may return it directly.
        return local_cdf_comp.result, True
      elif local_cdf_comp.result.is_lambda():
        # Case 3. We reduce to case 4 and pass through.
        local_cdf_comp = building_blocks.Lambda(
            local_cdf_comp.result.parameter_name,
            local_cdf_comp.result.parameter_type,
            building_blocks.Block(local_cdf_comp.locals,
                                  local_cdf_comp.result.result))
        # Reduce potential chain of blocks.
        local_cdf_comp, _ = tree_transformations.merge_chained_blocks(
            local_cdf_comp)
        # This fall-through is intended, since we have merged with case 4.
    if local_cdf_comp.is_lambda():
      # Case 4.
      repacked_block = _package_as_deduplicated_block(local_cdf_comp.result)
      tf_generated, _ = create_tensorflow_representing_block(repacked_block)
      tff_func = building_blocks.Lambda(local_cdf_comp.parameter_name,
                                        local_cdf_comp.parameter_type,
                                        tf_generated)
      tf_parser_callable = tree_to_cc_transformations.TFParser()
      tf_generated, _ = transformation_utils.transform_postorder(
          tff_func, tf_parser_callable)
    else:
      raise tree_transformations.TransformationError(
          'Unexpected structure encountered for functional computation in '
          'local call-dominant form: \n'
          f'{local_cdf_comp.formatted_representation()}')
  else:
    # The CFG for local call dominant tells us no lambdas or blocks may be
    # present under `comp` for non-functional types which can be represented in
    # TensorFlow (in particular, structures of functions are disallowed by this
    # restriction). So we may package as a block directly.
    repacked_block = _package_as_deduplicated_block(local_cdf_comp)
    tf_generated, _ = create_tensorflow_representing_block(repacked_block)
  return tf_generated, True


class RemoveDuplicatesAndApplyTransform(transformation_utils.TransformSpec):
  """Deduplicates before applying an interim transform, then repacks."""

  def __init__(self, comp: building_blocks.ComputationBuildingBlock,
               interim_transform_spec: transformation_utils.TransformSpec):
    """Constructs a new instance.

    Args:
      comp: Instance of `building_blocks.ComputationBuildingBlock` on which to
        apply the transform.
      interim_transform_spec: Instance of `transformation_utils.TransformSpec`
        whose `transform` method must take a `building_blocks.Struct` and return
        a named tuple type, to be applied after deduplication.

    Raises:
      TypeError: If types do not match.
      ValueError: If the `uri` has an unexpected value.
    """
    super().__init__()
    py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
    py_typecheck.check_type(interim_transform_spec,
                            transformation_utils.TransformSpec)
    self._name_generator = building_block_factory.unique_name_generator(comp)
    self._interim_transform = interim_transform_spec

  def should_transform(self, comp):
    return self._interim_transform.should_transform(comp) and comp.is_struct()

  def _construct_deduped_tuple_and_selection_map(self, comp):
    deduped_tuple = []
    selection_map = []
    for called_intrinsic in comp:
      index_in_deduped_tuple = None
      for idx, previous_called_intrinsic in enumerate(deduped_tuple):
        if tree_analysis.trees_equal(called_intrinsic,
                                     previous_called_intrinsic):
          index_in_deduped_tuple = idx
      if index_in_deduped_tuple is None:
        deduped_tuple.append(called_intrinsic)
        index_in_deduped_tuple = len(deduped_tuple) - 1
      selection_map.append(index_in_deduped_tuple)
    return deduped_tuple, selection_map

  def transform(self, comp):

    if not self.should_transform(comp):
      return comp, False

    deduped_tuple, selection_map = self._construct_deduped_tuple_and_selection_map(
        comp)
    transform_applied, _ = self._interim_transform.transform(
        building_blocks.Struct(deduped_tuple))
    transform_applied.type_signature.check_struct()
    if len(comp) == len(deduped_tuple):
      # Fall back if no optimization is made.
      return transform_applied, True
    lam_arg = building_blocks.Reference(
        next(self._name_generator), transform_applied.type_signature)
    replacement_tuple = []
    for i in selection_map:
      selected = building_blocks.Selection(lam_arg, index=i)
      replacement_tuple.append(selected)
    tup = building_blocks.Struct(replacement_tuple)
    lam = building_blocks.Lambda(lam_arg.name, lam_arg.type_signature, tup)
    return building_blocks.Call(lam, transform_applied), True


def optimize_tensorflow_graphs(comp, grappler_config_proto):
  """Performs any static optimization on TensorFlow subcomputations."""
  tf_optimizer = compiled_computation_transforms.TensorFlowOptimizer(
      grappler_config_proto)
  return transformation_utils.transform_postorder(comp, tf_optimizer.transform)


class TensorFlowGenerator(transformation_utils.TransformSpec):
  """TransformSpec which generates TensorFlow to represent local computation.

  Any TFF computation which declares as its parameters and return values only
  instances of `computation_types.SequenceType`,
  `computation_types.StructType`, and `computation_types.TensorType`s, and
  not capturing any references from an outer scope or containing any intrinsics,
  can be represented by a TensorFlow computation. This TransformSpec identifies
  computations such computations and generates a semantically equivalent
  TensorFlow computation.
  """

  def transform(self, local_computation):
    if not self.should_transform(local_computation):
      return local_computation, False
    generated_tf, _ = generate_tensorflow_for_local_computation(
        local_computation)
    if not (generated_tf.is_compiled_computation() or
            (generated_tf.is_call() and
             generated_tf.function.is_compiled_computation())):
      raise tree_transformations.TransformationError(
          'Failed to generate TensorFlow for a local function. '
          f'Generated a building block of type {type(generated_tf)} with '
          f'formatted rep {generated_tf.formatted_representation()}.')
    return generated_tf, True

  def should_transform(self, comp):
    if not (type_analysis.is_tensorflow_compatible_type(comp.type_signature) or
            (comp.type_signature.is_function() and
             type_analysis.is_tensorflow_compatible_type(
                 comp.type_signature.parameter) and
             type_analysis.is_tensorflow_compatible_type(
                 comp.type_signature.result))):
      return False
    elif comp.is_compiled_computation() or (
        comp.is_call() and comp.function.is_compiled_computation()):
      # These represent the final result of TF generation; no need to transform.
      return False
    unbound_refs = transformation_utils.get_map_of_unbound_references(
        comp)[comp]
    if unbound_refs:
      # We cannot represent these captures without further information.
      return False
    if tree_analysis.contains_types(
        comp, building_blocks.Intrinsic) or tree_analysis.contains_types(
            comp, building_blocks.Data):
      return False
    return True


def compile_local_computations_to_tensorflow(comp):
  """Compiles any fully specified local functions to a TensorFlow computation.

  This function walks the AST backing `comp` in a preorder manner, calling out
  to TF-generating functions when it encounters a subcomputation which can be
  represented in TensorFlow. The fact that this function walks preorder is
  extremely important to efficiency of the generated TensorFlow; if we instead
  traversed in a bottom-up fashion, we could potentially generate duplicated
  structures where such duplication is unnecessary.

  Consider for example a computation with structure:

    [TFComp()[0], TFComp()[1]]

  Due to its preorder walk, this function will call out to TF-generating
  utilities with the *entire* structure above; this structure still has enough
  information to detect that TFComp() should be equivalent in both invocations
  (at least, according to TFF's functional specification). If we traversed the
  AST backing `comp` in a bottom-up fashion, we would instead make separate
  calls to TF generation functions, resulting in a structure like:

    [TFComp0(), TFComp1()]

  where the graphs backing TFComp0 and TFComp1 share some graph substructure.
  TFF does not inspect the substructures of the graphs it generates, and would
  simply declare each of the above to be fully distinct invocations, and would
  require that each run when the resulting graph is invoked.

  We provide this function to ensure that callers of TFF's TF-generation
  utilities are usually shielded from such concerns.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` whose local
      computations we wish to compile to TensorFlow.

  Returns:
    A tuple whose first element represents an equivalent computation, but whose
    local computations are represented as TensorFlow graphs. The second element
    of this tuple is a Boolean indicating whether any transforamtion was made.
  """

  non_tf_compiled_comp_types = set()

  def _visit(comp):
    if comp.is_compiled_computation(
    ) and comp.proto.WhichOneof('computation') != 'tensorflow':
      non_tf_compiled_comp_types.add(comp.proto.WhichOneof('computation'))

  tree_analysis.visit_postorder(comp, _visit)
  if non_tf_compiled_comp_types:
    raise TypeError('Encountered non-TensorFlow compiled computation types {} '
                    'in argument {} to '
                    '`compile_local_computations_to_tensorflow`.'.format(
                        non_tf_compiled_comp_types,
                        comp.formatted_representation()))

  if comp.is_compiled_computation() or (
      comp.is_call() and comp.function.is_compiled_computation()):
    # These represent the final result of TF generation; no need to transform,
    # so we short-circuit here.
    return comp, False
  local_tf_generator = TensorFlowGenerator()
  return transformation_utils.transform_preorder(comp,
                                                 local_tf_generator.transform)


def transform_to_call_dominant(
    comp: building_blocks.ComputationBuildingBlock
) -> transformation_utils.TransformReturnType:
  """Normalizes computations into Call-Dominant Form.

  A computation is in call-dominant form if the following conditions are true:

  1. Every intrinsic which will be invoked to execute the computation appears
     as a top-level let binding (modulo an encapsulating global lambda).
  2. Each of these intrinsics is depended upon by the output. This implies in
     particular that any intrinsic which is not depended upon by the output is
     removed.
  3. All reference bindings have unique names.

  In an intermediate step, this function invokes
  `tree_transformations.resolve_higher_order_functions` in order to ensure that
  the function member of every `building_blocks.Call` must be either: a
  `building_blocks.CompiledComputation`; a `building_blocks.Intrinsic`;
  a `building_blocks.Lambda` with non-functional return type; a reference to
  a function bound as parameter to an uncalled `building_blocks.Lambda`;
  or a (possibly nested) selection from a reference to the parameter of
  an such an uncalled `building_blocks.Lambda`.

  Note that if no lambda takes a functional parameter, the final case in
  the enumeration above is additionally disallowed.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` to transform.

  Returns:
    A two-tuple, whose first element is a building block representing the same
    logic as `comp`, and whose second is a boolean indicating whether or not
    any transformations were in fact run.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)

  def _check_calls_are_concrete(comp):
    """Encodes condition for completeness of direct extraction of calls.

    After checking this condition, all functions which are semantically called
    (IE, functions which will be invoked eventually by running the computation)
    are called directly, and we can simply extract them by pattern-matching on
    `building_blocks.Call`.

    Args:
      comp: Instance of `building_blocks.ComputationBuildingBlock` to check for
        condition that functional argument of `Call` constructs contains only
        the enumeration in the top-level docstring.

    Raises:
      ValueError: If `comp` fails this condition.
    """
    symbol_tree = transformation_utils.SymbolTree(
        transformation_utils.ReferenceCounter)

    def _check_for_call_arguments(comp_to_check, symbol_tree):
      if not comp_to_check.is_call():
        return comp_to_check, False
      functional_arg = comp_to_check.function
      if functional_arg.is_compiled_computation(
      ) or functional_arg.is_intrinsic():
        return comp_to_check, False
      elif functional_arg.is_lambda():
        if type_analysis.contains(functional_arg.type_signature.result,
                                  lambda x: x.is_function()):
          raise ValueError('Called higher-order functions are disallowed in '
                           'transforming to call-dominant form, as they may '
                           'break the reliance on pattern-matching to extract '
                           'called intrinsics. Encountered a call to the'
                           'lambda {l} with type signature {t}.'.format(
                               l=functional_arg,
                               t=functional_arg.type_signature))
        return comp_to_check, False
      elif functional_arg.is_reference():
        # This case and the following handle the possibility that a lambda
        # declares a functional parameter, and this parameter is invoked in its
        # body.
        payload = symbol_tree.get_payload_with_name(functional_arg.name)
        if payload is None:
          return comp, False
        if payload.value is not None:
          raise ValueError('Called references which are not bound to lambda '
                           'parameters are disallowed in transforming to '
                           'call-dominant form, as they may break the reliance '
                           'on pattern-matching to extract called intrinsics. '
                           'Encountered a call to the reference {r}, which is '
                           'bound to the value {v} in this computation.'.format(
                               r=functional_arg, v=payload.value))
      elif functional_arg.is_selection():
        concrete_source = functional_arg.source
        while concrete_source.is_selection():
          concrete_source = concrete_source.source
        if concrete_source.is_reference():
          payload = symbol_tree.get_payload_with_name(concrete_source.name)
          if payload is None:
            return comp, False
          if payload.value is not None:
            raise ValueError('Called selections from references which are not '
                             'bound to lambda parameters are disallowed in '
                             'transforming to call-dominant form, as they may '
                             'break the reliance on pattern-matching to '
                             'extract called intrinsics. Encountered a call to '
                             'the reference {r}, which is bound to the value '
                             '{v} in this computation.'.format(
                                 r=functional_arg, v=payload.value))
          return comp, False
        else:
          raise ValueError('Called selections are only permitted in '
                           'transforming to call-comiunant form the case that '
                           'they select from lambda parameters; encountered a '
                           'call to selection {s}.'.format(s=functional_arg))
      else:
        raise ValueError('During transformation to call-dominant form, we rely '
                         'on the assumption that all called functions are '
                         'either: compiled computations; intrinsics; lambdas '
                         'with nonfuntional return types; or selections from '
                         'lambda parameters. Encountered the called function '
                         '{f} of type {t}.'.format(
                             f=functional_arg, t=type(functional_arg)))

    transformation_utils.transform_postorder_with_symbol_bindings(
        comp, _check_for_call_arguments, symbol_tree)

  def _inline_functions(comp):
    function_type_reference_names = []

    def _populate_function_type_ref_names(comp):
      if comp.is_reference() and comp.type_signature.is_function():
        function_type_reference_names.append(comp.name)
      return comp, False

    transformation_utils.transform_postorder(comp,
                                             _populate_function_type_ref_names)

    return tree_transformations.inline_block_locals(
        comp, variable_names=set(function_type_reference_names))

  def _extract_calls_and_blocks(comp):

    def _predicate(comp):
      return comp.is_call()

    block_extracter = tree_transformations.ExtractComputation(comp, _predicate)
    return transformation_utils.transform_postorder(comp,
                                                    block_extracter.transform)

  def _resolve_calls_to_concrete_functions(comp):
    """Removes symbol bindings which contain functional types."""

    comp, refs_renamed = tree_transformations.uniquify_reference_names(comp)
    comp, fns_resolved = tree_transformations.resolve_higher_order_functions(
        comp)
    comp, called_lambdas_replaced = tree_transformations.replace_called_lambda_with_block(
        comp)
    comp, selections_inlined = tree_transformations.inline_selections_from_tuple(
        comp)
    if fns_resolved or selections_inlined:
      comp, _ = tree_transformations.uniquify_reference_names(comp)
    comp, fns_inlined = _inline_functions(comp)
    comp, locals_removed = tree_transformations.remove_unused_block_locals(comp)

    modified = (
        refs_renamed or fns_resolved or called_lambdas_replaced or
        selections_inlined or fns_inlined or locals_removed)
    return comp, modified

  comp, modified = _resolve_calls_to_concrete_functions(comp)
  _check_calls_are_concrete(comp)

  for transform in [
      _extract_calls_and_blocks,
      # Extraction can leave some tuples packing references to clean up. Leaving
      # would not violate CDF, but we prefer to do this for cleanliness.
      tree_transformations.inline_selections_from_tuple,
      tree_transformations.merge_chained_blocks,
      tree_transformations.remove_duplicate_block_locals,
      tree_transformations.remove_unused_block_locals,
      tree_transformations.uniquify_reference_names,
  ]:
    comp, transformed = transform(comp)
    modified = modified or transformed
  return comp, modified


_NamedBinding = Tuple[str, building_blocks.ComputationBuildingBlock]


@attr.s
class _IntrinsicDependencies:
  uri_to_locals: Dict[str, List[_NamedBinding]] = attr.ib(
      factory=lambda: collections.defaultdict(list))
  locals_dependent_on_intrinsics: List[_NamedBinding] = attr.ib(factory=list)
  locals_not_dependent_on_intrinsics: List[_NamedBinding] = attr.ib(
      factory=list)


class _NonAlignableAlongIntrinsicError(ValueError):
  pass


def _compute_intrinsic_dependencies(
    intrinsic_uris: Set[str],
    parameter_name: str,
    locals_list: List[_NamedBinding],
    comp_repr,
) -> _IntrinsicDependencies:
  """Computes which locals have dependencies on which called intrinsics."""
  result = _IntrinsicDependencies()
  intrinsic_dependencies_for_ref: Dict[str, Set[str]] = {}
  # Start by marking `comp.parameter_name` as independent of intrinsics.
  intrinsic_dependencies_for_ref[parameter_name] = set()
  for local_name, local_value in locals_list:
    intrinsic_dependencies = set()

    def record_dependencies(subvalue):
      if subvalue.is_reference():
        if subvalue.name not in intrinsic_dependencies_for_ref:
          raise ValueError(
              f'Can\'t resolve {subvalue.name} in {[(name, value.compact_representation()) for name, value in locals_list]}\n'
              f'Current map {intrinsic_dependencies_for_ref}\n'
              f'Original comp: {comp_repr}\n')
        intrinsic_dependencies.update(  # pylint: disable=cell-var-from-loop
            intrinsic_dependencies_for_ref[subvalue.name])
      elif subvalue.is_lambda():
        # We treat the lambdas that appear in CDF (inside intrinsic invocations)
        # as though their parameters are independent of the rest of the
        # computation. Note that we're not careful about saving and then
        # restoring old variables here: this is okay because call-dominant form
        # guarantees unique variable names.
        intrinsic_dependencies_for_ref[subvalue.parameter_name] = set()
      elif subvalue.is_block():
        # Since we're in CDF, the only blocks inside the bodies of arguments
        # are within lambda arguments to intrinsics. We don't need to record
        # dependencies of these since they can't rely on the results of other
        # intrinsics.
        for subvalue_local_name, _ in subvalue.locals:
          intrinsic_dependencies_for_ref[subvalue_local_name] = set()

    tree_analysis.visit_preorder(local_value, record_dependencies)

    # All intrinsic calls are guaranteed to be top-level in call-dominant form.
    is_intrinsic_call = (
        local_value.is_call() and local_value.function.is_intrinsic() and
        local_value.function.uri in intrinsic_uris)
    if is_intrinsic_call:
      if intrinsic_dependencies:
        raise _NonAlignableAlongIntrinsicError(
            'Cannot force-align intrinsics:\n'
            f'Call to intrinsic `{local_value.function.uri}` depends '
            f'on calls to intrinsics:\n`{intrinsic_dependencies}`.')
      intrinsic_dependencies_for_ref[local_name] = set(
          [local_value.function.uri])
      result.uri_to_locals[local_value.function.uri].append(
          (local_name, local_value))
    else:
      intrinsic_dependencies_for_ref[local_name] = intrinsic_dependencies
      if intrinsic_dependencies:
        result.locals_dependent_on_intrinsics.append((local_name, local_value))
      else:
        result.locals_not_dependent_on_intrinsics.append(
            (local_name, local_value))
  return result


@attr.s
class _MergedIntrinsic:
  uri: str = attr.ib()
  args: building_blocks.ComputationBuildingBlock = attr.ib()
  return_type: computation_types.Type = attr.ib()
  unpack_to_locals: List[str] = attr.ib()


def _compute_merged_intrinsics(
    intrinsic_defaults: List[building_blocks.Call],
    uri_to_locals: Dict[str, List[_NamedBinding]],
    name_generator,
) -> List[_MergedIntrinsic]:
  """Computes a `_MergedIntrinsic` for each intrinsic in `intrinsic_defaults`.

  Args:
    intrinsic_defaults: A default call to each intrinsic URI to be merged. If no
      entry with this URI is present in `uri_to_locals`, the resulting
      `MergedIntrinsic` will describe only the provided default call.
    uri_to_locals: A mapping from intrinsic URI to locals (name + building_block
      pairs). The building block must be a `Call` to the intrinsic of the given
      URI. The name will be used to bind the result of the merged intrinsic via
      `_MergedIntrinsic.unpack_to_locals`.
    name_generator: A generator used to create unique names.

  Returns:
    A list of `_MergedIntrinsic`s describing, for each intrinsic, how to invoke
    the intrinsic exactly once and unpack the result.
  """
  results = []
  for default_call in intrinsic_defaults:
    uri = default_call.function.uri
    locals_for_uri = uri_to_locals[uri]
    if not locals_for_uri:
      results.append(
          _MergedIntrinsic(
              uri=uri,
              args=default_call.argument,
              return_type=default_call.type_signature,
              unpack_to_locals=[]))
    else:
      calls = [local[1] for local in locals_for_uri]
      result_placement = calls[0].type_signature.placement
      result_all_equal = calls[0].type_signature.all_equal
      for call in calls:
        if call.type_signature.all_equal != result_all_equal:
          raise ValueError('Encountered intrinsics to be merged with '
                           f'mismatched all_equal bits. Intrinsic of URI {uri} '
                           f'first call had all_equal bit {result_all_equal}, '
                           'encountered call with all_equal value '
                           f'{call.type_signature.all_equal}')
      return_type = computation_types.FederatedType(
          computation_types.StructType([
              (None, call.type_signature.member) for call in calls
          ]),
          placement=result_placement,
          all_equal=result_all_equal)
      abstract_parameter_type = default_call.function.intrinsic_def(
      ).type_signature.parameter
      results.append(
          _MergedIntrinsic(
              uri=uri,
              args=_merge_args(abstract_parameter_type,
                               [call.argument for call in calls],
                               name_generator),
              return_type=return_type,
              unpack_to_locals=[name for (name, _) in locals_for_uri],
          ))
  return results


def _merge_args(
    abstract_parameter_type,
    args: List[building_blocks.ComputationBuildingBlock],
    name_generator,
) -> building_blocks.ComputationBuildingBlock:
  """Merges the arguments of multiple function invocations into one.

  Args:
    abstract_parameter_type: The abstract parameter type specification for the
      function being invoked. This is used to determine whether any functional
      parameters accept multiple arguments.
    args: A list where each element contains the arguments to a single call.
    name_generator: A generator used to create unique names.

  Returns:
    A building block to use as the new (merged) argument.
  """
  if abstract_parameter_type.is_federated():
    zip_args = building_block_factory.create_federated_zip(
        building_blocks.Struct(args))
    # `create_federated_zip` introduces repeated names.
    zip_args, _ = tree_transformations.uniquify_reference_names(
        zip_args, name_generator)
    return zip_args
  if (abstract_parameter_type.is_tensor() or
      abstract_parameter_type.is_abstract()):
    return building_blocks.Struct([(None, arg) for arg in args])
  if abstract_parameter_type.is_function():
    # For functions, we must compose them differently depending on whether the
    # abstract function (from the intrinsic definition) takes more than one
    # parameter.
    #
    # If it does not, such as in the `fn` argument to `federated_map`, we can
    # simply select out the argument and call the result:
    # `(fn0(arg[0]), fn1(arg[1]), ..., fnN(arg[n]))`
    #
    # If it takes multiple arguments such as the `accumulate` argument to
    # `federated_aggregate`, we have to select out the individual arguments to
    # pass to each function:
    #
    # `(
    #   fn0(arg[0][0], arg[1][0]),
    #   fn1(arg[0][1], arg[1][1]),
    #   ...
    #   fnN(arg[0][n], arg[1][n]),
    # )`
    param_name = next(name_generator)
    if abstract_parameter_type.parameter.is_struct():
      num_args = len(abstract_parameter_type.parameter)
      parameter_types = [[] for i in range(num_args)]
      for arg in args:
        for i in range(num_args):
          parameter_types[i].append(arg.type_signature.parameter[i])
      param_type = computation_types.StructType(parameter_types)
      param_ref = building_blocks.Reference(param_name, param_type)
      calls = []
      for (n, fn) in enumerate(args):
        args_to_fn = []
        for i in range(num_args):
          args_to_fn.append(
              building_blocks.Selection(
                  building_blocks.Selection(param_ref, index=i), index=n))
        calls.append(
            building_blocks.Call(
                fn,
                building_blocks.Struct([(None, arg) for arg in args_to_fn])))
    else:
      param_type = computation_types.StructType(
          [arg.type_signature.parameter for arg in args])
      param_ref = building_blocks.Reference(param_name, param_type)
      calls = [
          building_blocks.Call(fn,
                               building_blocks.Selection(param_ref, index=n))
          for (n, fn) in enumerate(args)
      ]
    return building_blocks.Lambda(
        parameter_name=param_name,
        parameter_type=param_type,
        result=building_blocks.Struct([(None, call) for call in calls]))
  if abstract_parameter_type.is_struct():
    # Bind each argument to a name so that we can reference them multiple times.
    arg_locals = []
    arg_refs = []
    for arg in args:
      arg_name = next(name_generator)
      arg_locals.append((arg_name, arg))
      arg_refs.append(building_blocks.Reference(arg_name, arg.type_signature))
    merged_args = []
    for i in range(len(abstract_parameter_type)):
      ith_args = [building_blocks.Selection(ref, index=i) for ref in arg_refs]
      merged_args.append(
          _merge_args(abstract_parameter_type[i], ith_args, name_generator))
    return building_blocks.Block(
        arg_locals,
        building_blocks.Struct([(None, arg) for arg in merged_args]))
  raise TypeError(f'Cannot merge args of type: {abstract_parameter_type}')


def force_align_and_split_by_intrinsics(
    comp: building_blocks.Lambda,
    intrinsic_defaults: List[building_blocks.Call],
) -> Tuple[building_blocks.Lambda, building_blocks.Lambda]:
  """Divides `comp` into before-and-after of calls to one ore more intrinsics.

  The input computation `comp` must have the following properties:

  1. The computation `comp` is completely self-contained, i.e., there are no
     references to arguments introduced in a scope external to `comp`.

  2. `comp`'s return value must not contain uncalled lambdas.

  3. None of the calls to intrinsics in `intrinsic_defaults` may be
     within a lambda passed to another external function (intrinsic or
     compiled computation).

  4. No argument passed to an intrinsic in `intrinsic_defaults` may be
     dependent on the result of a call to an intrinsic in
     `intrinsic_uris_and_defaults`.

  5. All intrinsics in `intrinsic_defaults` must have "merge-able" arguments.
     Structs will be merged element-wise, federated values will be zipped, and
     functions will be composed:
       `f = lambda f1_arg, f2_arg: (f1(f1_arg), f2(f2_arg))`

  6. All intrinsics in `intrinsic_defaults` must return a single federated value
     whose member is the merged result of any merged calls, i.e.:
       `f(merged_arg).member = (f1(f1_arg).member, f2(f2_arg).member)`

  Under these conditions, (and assuming `comp` is a computation with non-`None`
  argument), this function will return two `building_blocks.Lambda`s `before`
  and `after` such that `comp` is semantically equivalent to the following
  expression*:

  ```
  (arg -> (let
    x=before(arg),
    y=intrinsic1(x[0]),
    z=intrinsic2(x[1]),
    ...
   in after(<arg, <y,z,...>>)))
  ```

  If `comp` is a no-arg computation, the returned computations will be
  equivalent (in the same sense as above) to:
  ```
  ( -> (let
    x=before(),
    y=intrinsic1(x[0]),
    z=intrinsic2(x[1]),
    ...
   in after(<y,z,...>)))
  ```

  *Note that these expressions may not be entirely equivalent under
  nondeterminism since there is no way in this case to handle computations in
  which `before` creates a random variable that is then used in `after`, since
  the only way for state to pass from `before` to `after` is for it to travel
  through one of the intrinsics.

  In this expression, there is only a single call to `intrinsic` that results
  from consolidating all occurrences of this intrinsic in the original `comp`.
  All logic in `comp` that produced inputs to any these intrinsic calls is now
  consolidated and jointly encapsulated in `before`, which produces a combined
  argument to all the original calls. All the remaining logic in `comp`,
  including that which consumed the outputs of the intrinsic calls, must have
  been encapsulated into `after`.

  If the original computation `comp` had type `(T -> U)`, then `before` and
  `after` would be `(T -> X)` and `(<T,Y> -> U)`, respectively, where `X` is
  the type of the argument to the single combined intrinsic call above. Note
  that `after` takes the output of the call to the intrinsic as well as the
  original argument to `comp`, as it may be dependent on both.

  Args:
    comp: The instance of `building_blocks.Lambda` that serves as the input to
      this transformation, as described above.
    intrinsic_defaults: A list of intrinsics with which to split the
      computation, provided as a list of `Call`s to insert if no intrinsic with
      a matching URI is found. Intrinsics in this list will be merged, and
      `comp` will be split across them.

  Returns:
    A pair of the form `(before, after)`, where each of `before` and `after`
    is a `building_blocks.ComputationBuildingBlock` instance that represents a
    part of the result as specified above.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(intrinsic_defaults, list)
  comp_repr = comp.compact_representation()

  # Flatten `comp` to call-dominant form so that we're working with just a
  # linear list of intrinsic calls with no indirection via tupling, selection,
  # blocks, called lambdas, or references.
  comp, _ = transform_to_call_dominant(comp)

  # CDF can potentially return blocks if there are variables not dependent on
  # the top-level parameter. We normalize these away.
  if not comp.is_lambda():
    comp.check_block()
    comp.result.check_lambda()
    if comp.result.result.is_block():
      additional_locals = comp.result.result.locals
      result = comp.result.result.result
    else:
      additional_locals = []
      result = comp.result.result
    # Note: without uniqueness, a local in `comp.locals` could potentially
    # shadow `comp.result.parameter_name`. However, `transform_to_call_dominant`
    # above ensure that names are unique, as it ends in a call to
    # `uniquify_reference_names`.
    comp = building_blocks.Lambda(
        comp.result.parameter_name, comp.result.parameter_type,
        building_blocks.Block(comp.locals + additional_locals, result))
  comp.check_lambda()

  # Simple computations with no intrinsic calls won't have a block.
  # Normalize these as well.
  if not comp.result.is_block():
    comp = building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                  building_blocks.Block([], comp.result))
  comp.result.check_block()

  name_generator = building_block_factory.unique_name_generator(comp)

  intrinsic_uris = set(call.function.uri for call in intrinsic_defaults)
  deps = _compute_intrinsic_dependencies(intrinsic_uris, comp.parameter_name,
                                         comp.result.locals, comp_repr)
  merged_intrinsics = _compute_merged_intrinsics(intrinsic_defaults,
                                                 deps.uri_to_locals,
                                                 name_generator)

  # Note: the outputs are labeled as `{uri}_param for convenience, e.g.
  # `federated_secure_sum_param: ...`.
  before = building_blocks.Lambda(
      comp.parameter_name, comp.parameter_type,
      building_blocks.Block(
          deps.locals_not_dependent_on_intrinsics,
          building_blocks.Struct([(f'{merged.uri}_param', merged.args)
                                  for merged in merged_intrinsics])))

  after_param_name = next(name_generator)
  if comp.parameter_type is not None:
    # TODO(b/147499373): If None-arguments were uniformly represented as empty
    # tuples, we would be able to avoid this (and related) ugly casing.
    after_param_type = computation_types.StructType([
        ('original_arg', comp.parameter_type),
        ('intrinsic_results',
         computation_types.StructType([(f'{merged.uri}_result',
                                        merged.return_type)
                                       for merged in merged_intrinsics])),
    ])
  else:
    after_param_type = computation_types.StructType([
        ('intrinsic_results',
         computation_types.StructType([(f'{merged.uri}_result',
                                        merged.return_type)
                                       for merged in merged_intrinsics])),
    ])
  after_param_ref = building_blocks.Reference(after_param_name,
                                              after_param_type)
  if comp.parameter_type is not None:
    original_arg_bindings = [
        (comp.parameter_name,
         building_blocks.Selection(after_param_ref, name='original_arg'))
    ]
  else:
    original_arg_bindings = []

  unzip_bindings = []
  for merged in merged_intrinsics:
    if merged.unpack_to_locals:
      intrinsic_result = building_blocks.Selection(
          building_blocks.Selection(after_param_ref, name='intrinsic_results'),
          name=f'{merged.uri}_result')
      select_param_type = intrinsic_result.type_signature.member
      for i, binding_name in enumerate(merged.unpack_to_locals):
        select_param_name = next(name_generator)
        select_param_ref = building_blocks.Reference(select_param_name,
                                                     select_param_type)
        selected = building_block_factory.create_federated_map_or_apply(
            building_blocks.Lambda(
                select_param_name, select_param_type,
                building_blocks.Selection(select_param_ref, index=i)),
            intrinsic_result)
        unzip_bindings.append((binding_name, selected))
  after = building_blocks.Lambda(
      after_param_name,
      after_param_type,
      building_blocks.Block(
          original_arg_bindings +
          # Note that we must duplicate `locals_not_dependent_on_intrinsics`
          # across both the `before` and `after` computations since both can
          # rely on them, and there's no way to plumb results from `before`
          # through to `after` except via one of the intrinsics being split
          # upon. In MapReduceForm, this limitation is caused by the fact that
          # `prepare` has no output which serves as an input to `report`.
          deps.locals_not_dependent_on_intrinsics + unzip_bindings +
          deps.locals_dependent_on_intrinsics,
          comp.result.result))
  try:
    tree_analysis.check_has_unique_names(before)
    tree_analysis.check_has_unique_names(after)
  except:
    raise ValueError(f'nonunique names in result of splitting\n{comp}')
  return before, after
