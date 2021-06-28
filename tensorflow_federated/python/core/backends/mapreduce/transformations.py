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
"""A library of TFF transformations specific to MapReduce backends.

In a nutshell, our overall strategy for compiling TFF computations for use with
MapReduce backends involves a three-stage process:

1. Bottom-up generic transformation phase. A sequence of transformations is
   applied that are common across backends. The goal of these is to reduce the
   complexity of the input (e.g., factoring out, merging elements, removing
   no-op code and other syntactic structures that are not needed). This phase
   is bottom-up, in the sense that it isn't driven by a specific target form of
   the output. It may consist or simple transformations applied repeatedly.

2. Top-down transformation phase specific to MapReduce backends. Our knowledge
   of the specific structure of the kind of processing these backends can
   support, captured in the definition of the "MapReduce form" and encoded in
   the definition of the `MapReduceForm` class in `map_reduce_form.py`, allows
   us to organize transformations in a manner that specifically supports the
   goal of converting a computation AST into the `MapReduceForm` eight-tuple
   of TensorFlow computations. This phase is top-down, in the sense that the
   converter from AST to `MapReduceForm` drives the process (e.g., it triggers
   what we may call a "force align" of certain communication operators, which
   may not make sense or be a valid and safe operation in general, but that is
   possible in the context of the kinds of computations that are convertible
   into a form consumable by MapReduce-like backends).

3. Final conversion from `MapReduceForm` into the form accepted by a given type
   of MapReduce backend, to be handled by the backend-specific code.

The second phase is essentially a form of "divide and conquer" that involves
two essential components:

a. The input computation is organized along the communication boundaries, to
   identify the parts that match the appropriate phases of processing (such as
   the prepare logic, or "everything before broadcast", client-side logic, and
   so on). It uses primitives such as finding all occurrences of a certain
   communication operator, restructuring the computation to align them together,
   what we refer to below as "force align", and calling generic transformations
   to merge the communication operators together.

b. The individual segments of the computation factored out in (a) above are now
   purely local processing (SERVER-only or CLIENTS-only), in which stronger
   assumptions can be made about structure and presence of various syntactic
   elements. Each such segment is reduced into a single section of TensorFlow.

In a nutshell, (a) above by means of force-align and similar mechanisms breaks
up the input computation into several regions on which (b) can be applied to
reduce it into a single TensorFlow section. The problem tackled in (b) is much
simpler than the problem of reducing the entire input computation, hence the
divide-and-conquer.
"""

import collections

from absl import logging

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class MapReduceFormCompilationError(Exception):
  pass


def check_extraction_result(before_extraction, extracted):
  """Checks parsing TFF to TF has constructed an object of correct type."""
  py_typecheck.check_type(before_extraction,
                          building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(extracted, building_blocks.ComputationBuildingBlock)
  if before_extraction.type_signature.is_function():
    if not extracted.is_compiled_computation():
      raise MapReduceFormCompilationError(
          'We expect to parse down to a `building_blocks.CompiledComputation`, '
          'since we have the functional type {} after unwrapping placement. '
          'Instead we have the computation {} of type {}'.format(
              before_extraction.type_signature, extracted,
              extracted.type_signature))
  else:
    if not extracted.is_call():
      raise MapReduceFormCompilationError(
          'We expect to parse down to a `building_blocks.Call`, since we have '
          'the non-functional type {} after unwrapping placement. Instead we '
          'have the computation {} of type {}'.format(
              before_extraction.type_signature, extracted,
              extracted.type_signature))
    if not extracted.function.is_compiled_computation():
      raise MapReduceFormCompilationError(
          'We expect to parse a computation of the non-functional type {} down '
          'to a called TensorFlow block. Instead we hav a call to the '
          'computation {} of type {}. This likely means that we the '
          'computation {} represents a case the Tff-to-TF parser is missing.'
          .format(before_extraction.type_signature, extracted.function,
                  extracted.function.type_signature, before_extraction))
  if not before_extraction.type_signature.is_equivalent_to(
      extracted.type_signature):
    raise MapReduceFormCompilationError(
        'We have extracted a TensorFlow block of the correct Python type, but '
        'incorrect TFF type signature. Before extraction, we had a TFF '
        'object of type signature {}, but after extraction, we have instead '
        'a TFF object of type signature {}'.format(
            before_extraction.type_signature, extracted.type_signature))


def consolidate_and_extract_local_processing(comp, grappler_config_proto):
  """Consolidates all the local processing in `comp`.

  The input computation `comp` must have the following properties:

  1. The output of `comp` may be of a federated type or unplaced. We refer to
     the placement `p` of that type as the placement of `comp`. There is no
     placement anywhere in the body of `comp` different than `p`. If `comp`
     is of a functional type, and has a parameter, the type of that parameter
     is a federated type placed at `p` as well, or unplaced if the result of
     the function is unplaced.

  2. The only intrinsics that may appear in the body of `comp` are those that
     manipulate data locally within the same placement. The exact set of these
     intrinsics will be gradually updated. At the moment, we support only the
     following:

     * Either `federated_apply` or `federated_map`, depending on whether `comp`
       is `SERVER`- or `CLIENTS`-placed. `federated_map_all_equal` is also
       allowed in the `CLIENTS`-placed case.

     * Either `federated_value_at_server` or `federated_value_at_clients`,
       likewise placement-dependent.

     * Either `federated_zip_at_server` or `federated_zip_at_clients`, again
       placement-dependent.

     Anything else, including `sequence_*` operators, should have been reduced
     already prior to calling this function.

  3. There are no lambdas in the body of `comp` except for `comp` itself being
     possibly a (top-level) lambda. All other lambdas must have been reduced.
     This requirement may eventually be relaxed by embedding lambda reducer into
     this helper method.

  4. If `comp` is of a functional type, it is either an instance of
     `building_blocks.CompiledComputation`, in which case there is nothing for
     us to do here, or a `building_blocks.Lambda`.

  5. There is at most one unbound reference under `comp`, and this is only
     allowed in the case that `comp` is not of a functional type.

  Aside from the intrinsics specified above, and the possibility of allowing
  lambdas, blocks, and references given the constraints above, the remaining
  constructs in `comp` include a combination of tuples, selections, calls, and
  sections of TensorFlow (as `CompiledComputation`s). This helper function does
  contain the logic to consolidate these constructs.

  The output of this transformation is always a single section of TensorFlow,
  which we henceforth refer to as `result`, the exact form of which depends on
  the placement of `comp` and the presence or absence of an argument.

  a. If there is no argument in `comp`, and `comp` is `SERVER`-placed, then
     the `result` is such that `comp` can be equivalently represented as:

     ```
     federated_value_at_server(result())
     ```

  b. If there is no argument in `comp`, and `comp` is `CLIENTS`-placed, then
     the `result` is such that `comp` can be equivalently represented as:

     ```
     federated_value_at_clients(result())
     ```

  c. If there is an argument in `comp`, and `comp` is `SERVER`-placed, then
     the `result` is such that `comp` can be equivalently represented as:

     ```
     (arg -> federated_apply(<result, arg>))
     ```

  d. If there is an argument in `comp`, and `comp` is `CLIENTS`-placed, then
     the `result` is such that `comp` can be equivalently represented as:

     ```
     (arg -> federated_map(<result, arg>))
     ```

  If the type of `comp` is `T@p` (thus `comp` is non-functional), the type of
  `result` is `T`, where `p` is the specific (concrete) placement of `comp`.

  If the type of `comp` is `(T@p -> U@p)`, then the type of `result` must be
  `(T -> U)`, where `p` is again a specific placement.

  Args:
    comp: An instance of `building_blocks.ComputationBuildingBlock` that serves
      as the input to this transformation, as described above.
    grappler_config_proto: An instance of `tf.compat.v1.ConfigProto` to
      configure Grappler graph optimization of the generated TensorFlow graph.
      If `grappler_config_proto` has
      `graph_options.rewrite_options.disable_meta_optimizer=True`, Grappler is
      bypassed.

  Returns:
    An instance of `building_blocks.CompiledComputation` that holds the
    TensorFlow section produced by this extraction step, as described above.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  comp.type_signature.check_function()
  unplaced, _ = tree_transformations.strip_placement(comp)
  extracted = parse_tff_to_tf(unplaced, grappler_config_proto)
  check_extraction_result(unplaced, extracted)
  return extracted


def parse_tff_to_tf(comp, grappler_config_proto):
  """Parses TFF construct `comp` into TensorFlow construct.

  Does not change the type signature of `comp`. Therefore may return either
  a `building_blocks.CompiledComputation` or a `building_blocks.Call` with no
  argument and function `building_blocks.CompiledComputation`.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` to parse down
      to a single TF block.
    grappler_config_proto: An instance of `tf.compat.v1.ConfigProto` to
      configure Grappler graph optimization of the generated TensorFlow graph.
      If `grappler_config_proto` has
      `graph_options.rewrite_options.disable_meta_optimizer=True`, Grappler is
      bypassed.

  Returns:
    The result of parsing TFF to TF. If successful, this is either a single
    `building_blocks.CompiledComputation`, or a call to one. If unsuccessful,
    there may be more TFF constructs still remaining. Notice it is not the job
    of this function, but rather its callers, to check that the result of this
    parse is as expected.
  """
  tf_parsed, _ = transformations.compile_local_computations_to_tensorflow(comp)

  # TODO(b/184883078): Remove this check and trust Grappler to disable itself
  # based on the `disable_meta_optimizer` config.
  should_skip_grappler = (
      grappler_config_proto.HasField('graph_options') and
      grappler_config_proto.graph_options.HasField('rewrite_options') and
      grappler_config_proto.graph_options.rewrite_options.disable_meta_optimizer
  )
  if not should_skip_grappler:
    logging.info('Using Grappler on `MapReduceForm` TensorFlow graphs.')
    tf_parsed, _ = transformations.optimize_tensorflow_graphs(
        tf_parsed, grappler_config_proto)

  return tf_parsed


def force_align_and_split_by_intrinsics(comp, uris):
  """Splits the logic of `comp` into before-and-after of calls to an intrinsic.

  The input computation `comp` must have the following properties:

  1. The computation `comp` is completely self-contained, i.e., there are no
     references to arguments introduced in a scope external to `comp`.

  2. There must be at least a single instance of a call to the intrinsic with
     the name specified as the second argument ('intrinsic`) above.

  3. None of the `intrinsic` calls are in the body of an uncalled lambda except
     for the `comp` itself which must be an uncalled lambda.

  4. The arguments fed to these intrinsics are likewise self-contained (have
     no external references), except for possibly a reference to the argument
     of `comp` in case `comp` is a lambda.

  5. There are no called intrinsics of identity `uri` nested within the body of
     another called intrinsic of identity `uri`.

  Under these conditions, this helper function must return a pair of building
  blocks `before` and `after` such that `comp` is semantically equivalent to
  the following expression:

  ```
  (arg -> (let
    x=before(arg),
    y=intrinsic1(x[0]),
    z=intrinsic2(x[1]),
    ...
   in after(<arg, <y,z,...>>)))
  ```

  In this expression, there is only a single call to `intrinsic` that results
  from consolidating all occurrences of this intrinsic in the original `comp`.
  All logic in `comp` that produced inputs to any these intrinsic calls is now
  consolidated and jointly encapsulated in `before`, which produces a combined
  argument to all the original calls. All the remaining logic in `comp`,
  including that which consumed the outputs of the intrinsic calls, must have
  been encapsulated into `after`.

  Additionally, if the `intrinsic` takes a tuple of arguments, then`before`
  should also be a `building_blocks.Struct`. Otherwise, both `before` and
  `after` are instances of `building_blocks.ComputationBuildingBlock`.

  If the original computation `comp` had type `(T -> U)`, then `before` and
  `after` would be `(T -> X)` and `(<T,Y> -> U)`, respectively, where `X` is
  the type of the argument to the single combined intrinsic call above, and
  `Y` is the type of the result. Note that `after` takes the output of the
  call to the intrinsic as well as the original argument `arg`, as it may be
  dependent on both.

  Whereas this helper function can be general, there are two special cases of
  interest: `federated_broadcast` and `federated_aggregate`.

  Args:
    comp: The instance of `building_blocks.Lambda` that serves as the input to
      this transformation, as described above.
    uris: A Python `list` of URI of intrinsics to force align and split.

  Returns:
    A pair of the form `(before, after)`, where each of `before` and `after`
    is a `building_blocks.ComputationBuildingBlock` instance that represents a
    part of the result as specified above.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(uris, list)
  for uri in uris:
    py_typecheck.check_type(uri, str)
  _check_contains_called_intrinsics(comp, uris)
  comp = _force_align_intrinsics_to_top_level_lambda(comp, uris)
  return _split_by_intrinsics_in_top_level_lambda(comp)


def _check_contains_called_intrinsics(comp, uri):
  """Checks if `comp` contains called intrinsics for the given `uri`.

  Args:
    comp: The `building_blocks.ComputationBuildingBlock` to test.
    uri: A Python `list` of URI of intrinsics.

  Returns:
    `True` if `comp` contains called intrinsics for the given `uri`, otherwise
    `False`.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(uri, list)
  for x in uri:
    py_typecheck.check_type(x, str)

  def get_uri_for_all_called_intrinsics(comp):
    existing_uri = set()

    def _update(comp):
      if building_block_analysis.is_called_intrinsic(comp, uri):
        existing_uri.add(comp.function.uri)
      return comp, False

    transformation_utils.transform_postorder(comp, _update)
    return existing_uri

  actual_uri = get_uri_for_all_called_intrinsics(comp)
  for expected_uri in uri:
    if expected_uri not in actual_uri:
      raise ValueError(
          'Expected an AST containing an intrinsic with the uri: {}, found '
          'none.'.format(expected_uri))


def _force_align_intrinsics_to_top_level_lambda(comp, uri):
  """Forcefully aligns `comp` by the intrinsics for the given `uri`.

  This function transforms `comp` by extracting, grouping, and potentially
  merging all the intrinsics for the given `uri`. The result of this
  transformation should contain exactly one instance of the intrinsic for the
  given `uri` that is bound only by the `parameter_name` of `comp`.

  Args:
    comp: The `building_blocks.Lambda` to align.
    uri: A Python `list` of URI of intrinsics.

  Returns:
    A new computation with the transformation applied or the original `comp`.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(uri, list)
  for x in uri:
    py_typecheck.check_type(x, str)

  comp, _ = tree_transformations.uniquify_reference_names(comp)
  if not _can_extract_intrinsics_to_top_level_lambda(comp, uri):
    comp, _ = tree_transformations.replace_called_lambda_with_block(comp)
  comp = _inline_block_variables_required_to_align_intrinsics(comp, uri)
  comp, modified = _extract_intrinsics_to_top_level_lambda(comp, uri)
  if modified:
    if len(uri) > 1:
      comp, _ = _group_by_intrinsics_in_top_level_lambda(comp)
    modified = False
    for intrinsic_uri in uri:
      comp, transform_modified = transformations.dedupe_and_merge_tuple_intrinsics(
          comp, intrinsic_uri)
      if transform_modified:
        # Required because merging called intrinsics invokes building block
        # factories that do not name references uniquely.
        comp, _ = tree_transformations.uniquify_reference_names(comp)
      modified = modified or transform_modified
    if modified:
      # Required because merging called intrinsics will nest the called
      # intrinsics such that they can no longer be split.
      comp, _ = _extract_intrinsics_to_top_level_lambda(comp, uri)
  return comp


def _get_called_intrinsics(comp, uri):
  """Returns a Python `list` of called intrinsics in `comp` for the given `uri`.

  Args:
    comp: The `building_blocks.ComputationBuildingBlock` to search.
    uri: A Python `list` of URI of intrinsics.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)
  py_typecheck.check_type(uri, list)
  for x in uri:
    py_typecheck.check_type(x, str)

  intrinsics = []

  def _update(comp):
    if building_block_analysis.is_called_intrinsic(comp, uri):
      intrinsics.append(comp)
    return comp, False

  transformation_utils.transform_postorder(comp, _update)
  return intrinsics


def _can_extract_intrinsics_to_top_level_lambda(comp, uri):
  """Tests if the intrinsic for the given `uri` can be extracted.

  This currently maps identically to: the called intrinsics we intend to hoist
  don't close over any intermediate variables. That is, any variables other than
  potentiall the top-level parameter the computation itself declares.

  Args:
    comp: The `building_blocks.Lambda` to test. The names of lambda parameters
      and block variables in `comp` must be unique.
    uri: A Python `list` of URI of intrinsics.

  Returns:
    `True` if the intrinsic can be extracted, otherwise `False`.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(uri, list)
  for x in uri:
    py_typecheck.check_type(x, str)
  tree_analysis.check_has_unique_names(comp)

  intrinsics = _get_called_intrinsics(comp, uri)
  return all(
      tree_analysis.contains_no_unbound_references(x, comp.parameter_name)
      for x in intrinsics)


def _inline_block_variables_required_to_align_intrinsics(comp, uri):
  """Inlines the variables required to align the intrinsic for the given `uri`.

  This function inlines only the block variables required to align an intrinsic,
  which is necessary because many transformations insert block variables that do
  not impact alignment and should not be inlined.

  Additionally, this function iteratively attempts to inline block variables a
  long as the intrinsic can not be extracted to the top level lambda. Meaning,
  that unbound references in variables that are inlined, will also be inlined.

  Args:
    comp: The `building_blocks.Lambda` to transform.
    uri: A Python `list` of URI of intrinsics.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    ValueError: If an there are unbound references, other than block variables,
      preventing an intrinsic with the given `uri` from being aligned.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(uri, list)
  for x in uri:
    py_typecheck.check_type(x, str)

  while not _can_extract_intrinsics_to_top_level_lambda(comp, uri):
    unbound_references = transformation_utils.get_map_of_unbound_references(
        comp)
    variable_names = set()
    intrinsics = _get_called_intrinsics(comp, uri)
    for intrinsic in intrinsics:
      names = unbound_references[intrinsic]
      names.discard(comp.parameter_name)
      variable_names.update(names)
    if not variable_names:
      raise tree_transformations.TransformationError(
          'Inlining `Block` variables has failed. Expected to find unbound '
          'references for called `Intrisic`s matching the URI: \'{}\', but '
          'none were found in the AST: \n{}'.format(
              uri, comp.formatted_representation()))
    comp, modified = tree_transformations.inline_block_locals(
        comp, variable_names=variable_names)
    if modified:
      comp, _ = tree_transformations.uniquify_reference_names(comp)
    else:
      raise tree_transformations.TransformationError(
          'Inlining `Block` variables has failed, this will result in an '
          'infinite loop. Expected to modify the AST by inlining the variable '
          'names: \'{}\', but no transformations to the AST: \n{}'.format(
              variable_names, comp.formatted_representation()))
  return comp


def _extract_intrinsics_to_top_level_lambda(comp, uri):
  r"""Extracts intrinsics in `comp` for the given `uri`.

  This transformation creates an AST such that all the called intrinsics for the
  given `uri` in body of the `building_blocks.Block` returned by the top level
  lambda have been extracted to the top level lambda and replaced by selections
  from a reference to the constructed variable.

                       Lambda
                       |
                       Block
                      /     \
        [x=Struct, ...]       Comp
           |
           [Call,                  Call                   Call]
           /    \                 /    \                 /    \
  Intrinsic      Comp    Intrinsic      Comp    Intrinsic      Comp

  The order of the extracted called intrinsics matches the order of `uri`.

  Note: if this function is passed an AST which contains nested called
  intrinsics, it will fail, as it will mutate the subcomputation containing
  the lower-level called intrinsics on the way back up the tree.

  Args:
    comp: The `building_blocks.Lambda` to transform. The names of lambda
      parameters and block variables in `comp` must be unique.
    uri: A URI of an intrinsic.

  Returns:
    A new computation with the transformation applied or the original `comp`.

  Raises:
    ValueError: If all the intrinsics for the given `uri` in `comp` are not
      exclusively bound by `comp`.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(uri, list)
  for x in uri:
    py_typecheck.check_type(x, str)
  tree_analysis.check_has_unique_names(comp)

  name_generator = building_block_factory.unique_name_generator(comp)

  intrinsics = _get_called_intrinsics(comp, uri)
  for intrinsic in intrinsics:
    if not tree_analysis.contains_no_unbound_references(intrinsic,
                                                        comp.parameter_name):
      raise ValueError(
          'Expected a computation which binds all the references in all the '
          'intrinsic with the uri: {}.'.format(uri))
  if len(intrinsics) > 1:
    order = {}
    for index, element in enumerate(uri):
      if element not in order:
        order[element] = index
    intrinsics = sorted(intrinsics, key=lambda x: order[x.function.uri])
    extracted_comp = building_blocks.Struct(intrinsics)
  else:
    extracted_comp = intrinsics[0]
  ref_name = next(name_generator)
  ref_type = computation_types.to_type(extracted_comp.type_signature)
  ref = building_blocks.Reference(ref_name, ref_type)

  def _should_transform(comp):
    return building_block_analysis.is_called_intrinsic(comp, uri)

  def _transform(comp):
    if not _should_transform(comp):
      return comp, False
    if len(intrinsics) > 1:
      index = intrinsics.index(comp)
      comp = building_blocks.Selection(ref, index=index)
      return comp, True
    else:
      return ref, True

  comp, _ = transformation_utils.transform_postorder(comp, _transform)
  comp = _insert_comp_in_top_level_lambda(
      comp, name=ref.name, comp_to_insert=extracted_comp)
  return comp, True


def _insert_comp_in_top_level_lambda(comp, name, comp_to_insert):
  """Inserts a computation into `comp` with the given `name`.

  Args:
    comp: The `building_blocks.Lambda` to transform. The names of lambda
      parameters and block variables in `comp` must be unique.
    name: The name to use.
    comp_to_insert: The `building_blocks.ComputationBuildingBlock` to insert.

  Returns:
    A new computation with the transformation applied or the original `comp`.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(name, str)
  py_typecheck.check_type(comp_to_insert,
                          building_blocks.ComputationBuildingBlock)
  tree_analysis.check_has_unique_names(comp)

  result = comp.result
  if result.is_block():
    variables = result.locals
    result = result.result
  else:
    variables = []
  variables.insert(0, (name, comp_to_insert))
  block = building_blocks.Block(variables, result)
  return building_blocks.Lambda(comp.parameter_name, comp.parameter_type, block)


def _group_by_intrinsics_in_top_level_lambda(comp):
  """Groups the intrinsics in the frist block local in the result of `comp`.

  This transformation creates an AST by replacing the tuple of called intrinsics
  found as the first local in the `building_blocks.Block` returned by the top
  level lambda with two new computations. The first computation is a tuple of
  tuples of called intrinsics, representing the original tuple of called
  intrinscis grouped by URI. The second computation is a tuple of selection from
  the first computations, representing original tuple of called intrinsics.

  It is necessary to group intrinsics before it is possible to merge them.

  Args:
    comp: The `building_blocks.Lambda` to transform.

  Returns:
    A `building_blocks.Lamda` that returns a `building_blocks.Block`, the first
    local variables of the retunred `building_blocks.Block` will be a tuple of
    tuples of called intrinsics representing the original tuple of called
    intrinscis grouped by URI.

  Raises:
    ValueError: If the first local in the `building_blocks.Block` referenced by
      the top level lambda is not a `building_blocks.Struct` of called
      intrinsics.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(comp.result, building_blocks.Block)
  tree_analysis.check_has_unique_names(comp)

  name_generator = building_block_factory.unique_name_generator(comp)

  name, first_local = comp.result.locals[0]
  py_typecheck.check_type(first_local, building_blocks.Struct)
  for element in first_local:
    if not building_block_analysis.is_called_intrinsic(element):
      raise ValueError(
          'Expected all the elements of the `building_blocks.Struct` to be '
          'called intrinsics, but found: \n{}'.format(element))

  # Create collections of data describing how to pack and unpack the intrinsics
  # into groups by their URI.
  #
  # packed_keys is a list of unique URI ordered by occurrence in the original
  #   tuple of called intrinsics.
  # packed_groups is a `collections.OrderedDict` where each key is a URI to
  #   group by and each value is a list of intrinsics with that URI.
  # packed_indexes is a list of tuples where each tuple contains two indexes:
  #   the first index in the tuple is the index of the group that the intrinsic
  #   was packed into; the second index in the tuple is the index of the
  #   intrinsic in that group that the intrinsic was packed into; the index of
  #   the tuple in packed_indexes corresponds to the index of the intrinsic in
  #   the list of intrinsics that are beging grouped. Therefore, packed_indexes
  #   represents an implicit mapping of packed indexes, keyed by unpacked index.
  packed_keys = []
  for called_intrinsic in first_local:
    uri = called_intrinsic.function.uri
    if uri not in packed_keys:
      packed_keys.append(uri)
  # If there are no duplicates, return early.
  if len(packed_keys) == len(first_local):
    return comp, False
  packed_groups = collections.OrderedDict([(x, []) for x in packed_keys])
  packed_indexes = []
  for called_intrinsic in first_local:
    packed_group = packed_groups[called_intrinsic.function.uri]
    packed_group.append(called_intrinsic)
    packed_indexes.append((
        packed_keys.index(called_intrinsic.function.uri),
        len(packed_group) - 1,
    ))

  packed_elements = []
  for called_intrinsics in packed_groups.values():
    if len(called_intrinsics) > 1:
      element = building_blocks.Struct(called_intrinsics)
    else:
      element = called_intrinsics[0]
    packed_elements.append(element)
  packed_comp = building_blocks.Struct(packed_elements)

  packed_ref_name = next(name_generator)
  packed_ref_type = computation_types.to_type(packed_comp.type_signature)
  packed_ref = building_blocks.Reference(packed_ref_name, packed_ref_type)

  unpacked_elements = []
  for indexes in packed_indexes:
    group_index = indexes[0]
    sel = building_blocks.Selection(packed_ref, index=group_index)
    uri = packed_keys[group_index]
    called_intrinsics = packed_groups[uri]
    if len(called_intrinsics) > 1:
      intrinsic_index = indexes[1]
      sel = building_blocks.Selection(sel, index=intrinsic_index)
    unpacked_elements.append(sel)
  unpacked_comp = building_blocks.Struct(unpacked_elements)

  variables = comp.result.locals
  variables[0] = (name, unpacked_comp)
  variables.insert(0, (packed_ref_name, packed_comp))
  block = building_blocks.Block(variables, comp.result.result)
  fn = building_blocks.Lambda(comp.parameter_name, comp.parameter_type, block)
  return fn, True


def _split_by_intrinsics_in_top_level_lambda(comp):
  """Splits by the intrinsics in the frist block local in the result of `comp`.

  This function splits `comp` into two computations `before` and `after` the
  called intrinsic or tuple of called intrinsics found as the first local in the
  `building_blocks.Block` returned by the top level lambda; and returns a Python
  tuple representing the pair of `before` and `after` computations.

  Args:
    comp: The `building_blocks.Lambda` to split.

  Returns:
    A pair of `building_blocks.ComputationBuildingBlock`s.

  Raises:
    ValueError: If the first local in the `building_blocks.Block` referenced by
      the top level lambda is not a called intrincs or a
      `building_blocks.Struct` of called intrinsics.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(comp.result, building_blocks.Block)
  tree_analysis.check_has_unique_names(comp)

  name_generator = building_block_factory.unique_name_generator(comp)

  name, first_local = comp.result.locals[0]
  if building_block_analysis.is_called_intrinsic(first_local):
    result = first_local.argument
  elif first_local.is_struct():
    elements = []
    for element in first_local:
      if not building_block_analysis.is_called_intrinsic(element):
        raise ValueError(
            'Expected all the elements of the `building_blocks.Struct` to be '
            'called intrinsics, but found: \n{}'.format(element))
      elements.append(element.argument)
    result = building_blocks.Struct(elements)
  else:
    raise ValueError(
        'Expected either a called intrinsic or a `building_blocks.Struct` of '
        'called intrinsics, but found: \n{}'.format(first_local))

  before = building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                  result)

  ref_name = next(name_generator)
  ref_type = computation_types.StructType(
      (comp.parameter_type, first_local.type_signature))
  ref = building_blocks.Reference(ref_name, ref_type)
  sel_after_arg_1 = building_blocks.Selection(ref, index=0)
  sel_after_arg_2 = building_blocks.Selection(ref, index=1)

  variables = comp.result.locals
  variables[0] = (name, sel_after_arg_2)
  variables.insert(0, (comp.parameter_name, sel_after_arg_1))
  block = building_blocks.Block(variables, comp.result.result)
  after = building_blocks.Lambda(ref.name, ref.type_signature, block)
  return before, after


def select_output_from_lambda(comp, indices):
  """Constructs a new function with result of selecting `indices` from `comp`.

  Args:
    comp: Instance of `building_blocks.Lambda` of result type `tff.StructType`
      from which we wish to select `indices`. Notice that this named tuple type
      must have elements of federated type.
    indices: Instance of `int`, `list`, or `tuple`, specifying the indices we
      wish to select from the result of `comp`. If `indices` is an `int`, the
      result of the returned `comp` will be of type at index `indices` in
      `comp.type_signature.result`. If `indices` is a `list` or `tuple`, the
      result type will be a `tff.StructType` wrapping the specified selections.

  Returns:
    A transformed version of `comp` with result value the selection from the
    result of `comp` specified by `indices`.
  """
  py_typecheck.check_type(comp, building_blocks.Lambda)
  py_typecheck.check_type(comp.type_signature.result,
                          computation_types.StructType)
  py_typecheck.check_type(indices, (int, tuple, list))

  def _create_selected_output(comp, index, is_struct_opt):
    if is_struct_opt:
      return comp[index]
    else:
      return building_blocks.Selection(comp, index=index)

  result_tuple = comp.result
  tuple_opt = result_tuple.is_struct()
  elements = []
  if isinstance(indices, (tuple, list)):
    for x in indices:
      if isinstance(x, (tuple, list)):
        selected_output = result_tuple
        for y in x:
          tuple_opt = selected_output.is_struct()
          selected_output = _create_selected_output(selected_output, y,
                                                    tuple_opt)
      else:
        selected_output = _create_selected_output(result_tuple, x, tuple_opt)
      elements.append(selected_output)
    result = building_blocks.Struct(elements)
  else:
    if tuple_opt:
      result = result_tuple[indices]
    else:
      result = building_blocks.Selection(result_tuple, index=indices)
  return building_blocks.Lambda(comp.parameter_name, comp.parameter_type,
                                result)


def concatenate_function_outputs(first_function, second_function):
  """Constructs a new function concatenating the outputs of its arguments.

  Assumes that `first_function` and `second_function` already have unique
  names, and have declared parameters of the same type. The constructed
  function will bind its parameter to each of the parameters of
  `first_function` and `second_function`, and return the result of executing
  these functions in parallel and concatenating the outputs in a tuple.

  Args:
    first_function: Instance of `building_blocks.Lambda` whose result we wish to
      concatenate with the result of `second_function`.
    second_function: Instance of `building_blocks.Lambda` whose result we wish
      to concatenate with the result of `first_function`.

  Returns:
    A new instance of `building_blocks.Lambda` with unique names representing
    the computation described above.

  Raises:
    TypeError: If the arguments are not instances of `building_blocks.Lambda`,
    or declare parameters of different types.
  """

  py_typecheck.check_type(first_function, building_blocks.Lambda)
  py_typecheck.check_type(second_function, building_blocks.Lambda)
  tree_analysis.check_has_unique_names(first_function)
  tree_analysis.check_has_unique_names(second_function)

  if first_function.parameter_type != second_function.parameter_type:
    raise TypeError('Must pass two functions which declare the same parameter '
                    'type to `concatenate_function_outputs`; you have passed '
                    'one function which declared a parameter of type {}, and '
                    'another which declares a parameter of type {}'.format(
                        first_function.type_signature,
                        second_function.type_signature))

  def _rename_first_function_arg(comp):
    if comp.is_reference() and comp.name == first_function.parameter_name:
      if comp.type_signature != second_function.parameter_type:
        raise AssertionError('{}, {}'.format(comp.type_signature,
                                             second_function.parameter_type))
      return building_blocks.Reference(second_function.parameter_name,
                                       comp.type_signature), True
    return comp, False

  first_function, _ = transformation_utils.transform_postorder(
      first_function, _rename_first_function_arg)

  concatenated_function = building_blocks.Lambda(
      second_function.parameter_name, second_function.parameter_type,
      building_blocks.Struct([first_function.result, second_function.result]))

  renamed, _ = tree_transformations.uniquify_reference_names(
      concatenated_function)

  return renamed


def normalize_all_equal_bit(comp):
  """Normalizes the all equal bits under `comp`.

  For any computation of `tff.FederatedType`, we rely on uniformity of the
  `all_equal` bit to compile down to MapReduce form. For example, the values
  processed on the clients can only be accessed through a `federated_zip`,
  which produces a value with its `all_equal` bit set to `False`. Therefore
  any client processing cannot rely on processing values with `True`
  `all_equal` bits. This function forces all `tff.CLIENTS`-placed values
  to have `all_equal` bits set to `False`, while all `tff.SERVER`-placed
  values will have `all_equal` bits set to `True`.

  Notice that `normalize_all_equal_bit` relies on the "normal" all_equal bit
  being inserted in the construction of a new `tff.FederatedType`; the
  constructor by default sets this bit to match the pattern above, so we simply
  ask it to create a new `tff.FederatedType` for us.

  Args:
    comp: Instance of `building_blocks.ComputationBuildingBlock` whose placed
      values will have their `all_equal` bits normalized.

  Returns:
    A modified version of `comp` with all `tff.CLIENTS`-placed values having
    `all_equal False`, and all `tff.SERVER`-placed values having
    `all_equal True`.
  """
  py_typecheck.check_type(comp, building_blocks.ComputationBuildingBlock)

  def _normalize_reference_bit(comp):
    if not comp.type_signature.is_federated():
      return comp, False
    return building_blocks.Reference(
        comp.name,
        computation_types.FederatedType(comp.type_signature.member,
                                        comp.type_signature.placement)), True

  def _normalize_lambda_bit(comp):
    if not comp.parameter_type.is_federated():
      return comp, False
    return building_blocks.Lambda(
        comp.parameter_name,
        computation_types.FederatedType(comp.parameter_type.member,
                                        comp.parameter_type.placement),
        comp.result), True

  def _normalize_intrinsic_bit(comp):
    """Replaces federated map all equal with federated map."""
    if comp.uri != intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri:
      return comp, False
    parameter_type = [
        comp.type_signature.parameter[0],
        computation_types.FederatedType(comp.type_signature.parameter[1].member,
                                        placements.CLIENTS)
    ]
    intrinsic_type = computation_types.FunctionType(
        parameter_type,
        computation_types.FederatedType(comp.type_signature.result.member,
                                        placements.CLIENTS))
    new_intrinsic = building_blocks.Intrinsic(intrinsic_defs.FEDERATED_MAP.uri,
                                              intrinsic_type)
    return new_intrinsic, True

  def _transform_switch(comp):
    if comp.is_reference():
      return _normalize_reference_bit(comp)
    elif comp.is_lambda():
      return _normalize_lambda_bit(comp)
    elif comp.is_intrinsic():
      return _normalize_intrinsic_bit(comp)
    return comp, False

  return transformation_utils.transform_postorder(comp, _transform_switch)[0]
