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
"""A library of composite transformation functions.

A composite transformation is one that applies multiple atomic transformation to
an AST either pointwise or serially.
"""

from typing import Mapping

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


def dedupe_and_merge_tuple_intrinsics(comp, uri):
  r"""Merges tuples of called intrinsics into one called intrinsic."""

  # TODO(b/147359721): The application of the function below is a workaround to
  # a known pattern preventing TFF from deduplicating, effectively because tree
  # equality won't determine that [a, a][0] and [a, a][1] are actually the same
  # thing. A fuller fix is planned, but requires increasing the invariants
  # respected further up the TFF compilation pipelines. That is, in order to
  # reason about sufficiency of our ability to detect duplicates at this layer,
  # we would very much prefer to be operating in the subset of TFF effectively
  # representing local computation.

  def _remove_selection_from_block_holding_tuple(comp):
    """Reduces selection from a block holding a tuple."""
    if (comp.is_selection() and comp.source.is_block() and
        comp.source.result.is_struct()):
      if comp.index is None:
        names = [
            x[0] for x in structure.iter_elements(comp.source.type_signature)
        ]
        index = names.index(comp.name)
      else:
        index = comp.index
      return building_blocks.Block(comp.source.locals,
                                   comp.source.result[index]), True
    return comp, False

  comp, _ = transformation_utils.transform_postorder(
      comp, _remove_selection_from_block_holding_tuple)
  transform_spec = tree_transformations.MergeTupleIntrinsics(comp, uri)
  dedupe_and_merger = RemoveDuplicatesAndApplyTransform(comp, transform_spec)
  return transformation_utils.transform_postorder(comp,
                                                  dedupe_and_merger.transform)


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
