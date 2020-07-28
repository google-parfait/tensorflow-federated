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

from absl import logging

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl import tree_to_cc_transformations
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import compiled_computation_transforms
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.types import type_analysis


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


def remove_called_lambdas_and_blocks(comp):
  """Removes any called lambdas and blocks from `comp`.

  This function first resolves any higher-order functions, so that replacing
  called lambdas with blocks and then inlining the block locals cannot result
  in more called lambdas. It then performs this sequence of transformations,
  taking care to inline selections from tuples before inlining the rest of
  the block locals to prevent possible combinatorial growth of the generated
  AST.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` from which we
      want to remove called lambdas and blocks.

  Returns:
    A transformed version of `comp` which has no called lambdas or blocks, and
    no extraneous selections from tuples.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  comp, names_uniquified = tree_transformations.uniquify_reference_names(comp)
  comp, fns_resolved = tree_transformations.resolve_higher_order_functions(comp)
  comp, lambdas_replaced = tree_transformations.replace_called_lambda_with_block(
      comp)
  if fns_resolved or lambdas_replaced:
    comp, _ = tree_transformations.uniquify_reference_names(comp)
  comp, sels_removed = tree_transformations.inline_selections_from_tuple(comp)
  if sels_removed:
    comp, _ = tree_transformations.uniquify_reference_names(comp)
  comp, locals_inlined = tree_transformations.inline_block_locals(comp)
  modified = names_uniquified or fns_resolved or lambdas_replaced or sels_removed or locals_inlined
  return comp, modified


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


def remove_duplicate_called_graphs(comp):
  """Deduplicates called graphs for a subset of TFF AST constructs.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` whose called
      graphs we wish to deduplicate, according to `tree_analysis.trees_equal`.
      For `comp` to be eligible here, it must be either a lambda itself whose
      body contains no lambdas or blocks, or another computation containing no
      lambdas or blocks. This restriction is necessary because
      `remove_duplicate_called_graphs` makes no effort to ensure that it is not
      pulling references out of their defining scope, except for the case where
      `comp` is a lambda itself. This function exits early and logs a warning if
      this assumption is violated. Additionally, `comp` must contain only
      computations which can be represented in TensorFlow, IE, satisfy the type
      restriction in `type_analysis.is_tensorflow_compatible_type`.

  Returns:
    Either a called instance of `building_blocks.CompiledComputation` or a
    `building_blocks.CompiledComputation` itself, depending on whether `comp`
    is of non-functional or functional type respectively. Additionally, returns
    a boolean to match the `transformation_utils.TransformSpec` pattern.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  tree_analysis.check_has_unique_names(comp)
  name_generator = building_block_factory.unique_name_generator(comp)
  if comp.is_lambda():
    comp_to_check = comp.result
  else:
    comp_to_check = comp
  if tree_analysis.contains_types(comp_to_check, (
      building_blocks.Block,
      building_blocks.Lambda,
  )):
    logging.warning(
        'The preprocessors have failed to remove called lambdas '
        'and blocks; falling back to less efficient, but '
        'guaranteed, TensorFlow generation with computation %s.', comp)
    return comp, False

  leaf_called_graphs = []

  def _pack_called_graphs_into_block(inner_comp):
    """Packs deduplicated bindings to called graphs in `leaf_called_graphs`."""
    if inner_comp.is_call() and inner_comp.function.is_compiled_computation():
      for (name, x) in leaf_called_graphs:
        if tree_analysis.trees_equal(x, inner_comp):
          return building_blocks.Reference(name,
                                           inner_comp.type_signature), True
      new_name = next(name_generator)
      leaf_called_graphs.append((new_name, inner_comp))
      return building_blocks.Reference(new_name,
                                       inner_comp.type_signature), True

    return inner_comp, False

  if comp.is_lambda():
    transformed_result, _ = transformation_utils.transform_postorder(
        comp.result, _pack_called_graphs_into_block)
    packed_into_block = building_blocks.Block(leaf_called_graphs,
                                              transformed_result)
    parsed, _ = create_tensorflow_representing_block(packed_into_block)
    tff_func = building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                      parsed)
    tf_parser_callable = tree_to_cc_transformations.TFParser()
    comp, _ = tree_transformations.insert_called_tf_identity_at_leaves(tff_func)
    tf_generated, _ = transformation_utils.transform_postorder(
        comp, tf_parser_callable)
  else:
    transformed_result, _ = transformation_utils.transform_postorder(
        comp, _pack_called_graphs_into_block)
    packed_into_block = building_blocks.Block(leaf_called_graphs,
                                              transformed_result)
    tf_generated, _ = create_tensorflow_representing_block(packed_into_block)
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

  def __init__(self):
    self._naive_tf_parser = tree_to_cc_transformations.TFParser()

  def transform(self, local_function):
    if not self.should_transform(local_function):
      return local_function, False
    refs_removed, _ = remove_called_lambdas_and_blocks(local_function)
    parsed_to_tf, _ = remove_duplicate_called_graphs(refs_removed)
    if parsed_to_tf.is_compiled_computation() or (
        parsed_to_tf.is_call() and
        parsed_to_tf.function.is_compiled_computation()):
      return parsed_to_tf, True
    # TODO(b/146430051): We should only end up in this case if
    # `remove_called_lambdas_and_blocks` above is in its failure mode, IE,
    # failing to resolve references due to too-deep indirection; we should
    # remove this extra case and simply raise if we fail here when we fix the
    # attached bug.
    called_graphs_inserted, _ = tree_transformations.insert_called_tf_identity_at_leaves(
        parsed_to_tf)
    compiled_comp, _ = transformation_utils.transform_postorder(
        called_graphs_inserted, self._naive_tf_parser)
    return compiled_comp, True

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
    if tree_analysis.contains_types(comp, building_blocks.Intrinsic):
      return False
    return True


def compile_local_computation_to_tensorflow(comp):
  """Compiles any fully specified local function to a TensorFlow computation."""
  if comp.is_compiled_computation() or (
      comp.is_call() and comp.function.is_compiled_computation()):
    # These represent the final result of TF generation; no need to transform,
    # so we short-circuit here.
    return comp, False
  local_tf_generator = TensorFlowGenerator()
  transformed, modified = transformation_utils.transform_preorder(
      comp, local_tf_generator.transform)
  return transformed, modified


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
    if selections_inlined:
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
