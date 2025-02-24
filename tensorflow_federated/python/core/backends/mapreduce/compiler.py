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
"""A compiler for the mapreduce backend.

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
   the definition of the `MapReduceForm` class in `forms.py`, allows
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

from absl import logging
import federated_language
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_backend import compiled_computation_transformations
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_tree_transformations
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_transformations


class MapReduceFormCompilationError(Exception):
  pass


def check_extraction_result(before_extraction, extracted):
  """Checks parsing TFF to TF has constructed an object of correct type."""
  py_typecheck.check_type(
      before_extraction, federated_language.framework.ComputationBuildingBlock
  )
  py_typecheck.check_type(
      extracted, federated_language.framework.ComputationBuildingBlock
  )
  if isinstance(
      before_extraction.type_signature, federated_language.FunctionType
  ):
    if not isinstance(
        extracted, federated_language.framework.CompiledComputation
    ):
      raise MapReduceFormCompilationError(
          'We expect to parse down to a'
          ' `federated_language.framework.CompiledComputation`, since we have'
          ' the functional type {} after unwrapping placement. Instead we have'
          ' the computation {} of type {}'.format(
              before_extraction.type_signature,
              extracted,
              extracted.type_signature,
          )
      )
  else:
    if not isinstance(extracted, federated_language.framework.Call):
      raise MapReduceFormCompilationError(
          'We expect to parse down to a `federated_language.framework.Call`,'
          ' since we have the non-functional type {} after unwrapping'
          ' placement. Instead we have the computation {} of type {}'.format(
              before_extraction.type_signature,
              extracted,
              extracted.type_signature,
          )
      )
    if not isinstance(
        extracted.function, federated_language.framework.CompiledComputation
    ):
      raise MapReduceFormCompilationError(
          'We expect to parse a computation of the non-functional type {} down '
          'to a called TensorFlow block. Instead we hav a call to the '
          'computation {} of type {}. This likely means that we the '
          'computation {} represents a case the Tff-to-TF parser is missing.'
          .format(
              before_extraction.type_signature,
              extracted.function,
              extracted.function.type_signature,
              before_extraction,
          )
      )
  if not before_extraction.type_signature.is_assignable_from(
      extracted.type_signature
  ):
    # In some situations, TF may can statically determine more type information
    # after TFF has coalesced computations (example: calling an identity
    # function of type (int32[?] -> int32[?]) on an argument of type int32[1]).
    # This one-way assignability check allows the TFF compiler to return a TF
    # computation with a more specific type if possible.
    raise MapReduceFormCompilationError(
        'We have extracted a TensorFlow block of the correct Python type, but '
        'incorrect TFF type signature. Before extraction, we had a TFF '
        'object of type signature {}, but after extraction, we have instead '
        'a TFF object of type signature {}'.format(
            before_extraction.type_signature, extracted.type_signature
        )
    )


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

     * GENERIC_* intrinsic calls.

     Anything else, including `sequence_*` operators, should have been reduced
     already prior to calling this function.

  3. There are no lambdas in the body of `comp` except for `comp` itself being
     possibly a (top-level) lambda. All other lambdas must have been reduced.
     This requirement may eventually be relaxed by embedding lambda reducer into
     this helper method.

  4. If `comp` is of a functional type, it is either an instance of
     `federated_language.framework.CompiledComputation`, in which case there is
     nothing for
     us to do here, or a `federated_language.framework.Lambda`.

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
    comp: An instance of `federated_language.framework.ComputationBuildingBlock`
      that serves as the input to this transformation, as described above.
    grappler_config_proto: An instance of `tf.compat.v1.ConfigProto` to
      configure Grappler graph optimization of the generated TensorFlow graph.
      If `grappler_config_proto` has
      `graph_options.rewrite_options.disable_meta_optimizer=True`, Grappler is
      bypassed.

  Returns:
    An instance of `federated_language.framework.CompiledComputation` that holds
    the
    TensorFlow section produced by this extraction step, as described above.
  """
  py_typecheck.check_type(
      comp, federated_language.framework.ComputationBuildingBlock
  )
  if not isinstance(comp.type_signature, federated_language.FunctionType):
    raise ValueError(
        'Expected a `federated_language.FunctionType`, found'
        f' {comp.type_signature}.'
    )
  # Replace any GENERIC_* intrinsic calls with TF function calls.
  comp, _ = tensorflow_tree_transformations.replace_intrinsics_with_bodies(comp)
  # Drop any unused subcomputations which may reference placements different
  # from the result.
  simplified = transformations.to_call_dominant(comp)
  unplaced, _ = tree_transformations.strip_placement(simplified)
  extracted = parse_tff_to_tf(unplaced, grappler_config_proto)
  check_extraction_result(unplaced, extracted)
  return extracted


def unpack_compiled_computations(
    comp: federated_language.framework.ComputationBuildingBlock,
) -> federated_language.framework.ComputationBuildingBlock:
  """Deserializes compiled computations into building blocks where possible."""

  def _unpack(subcomp):
    if not isinstance(
        subcomp, federated_language.framework.CompiledComputation
    ):
      return subcomp, False
    kind = subcomp.to_proto().WhichOneof('computation')
    if kind == 'tensorflow' or kind == 'xla':
      return subcomp, False
    return (
        federated_language.framework.ComputationBuildingBlock.from_proto(
            subcomp.to_proto()
        ),
        True,
    )

  comp, _ = federated_language.framework.transform_postorder(comp, _unpack)
  return comp


class XlaToTensorFlowError(ValueError):
  """An error indicating an attempt to compile XLA code to TensorFlow."""


class ExternalBlockToTensorFlowError(ValueError):
  """An error indicating an attempt to compile external blocks to TensorFlow."""


def _evaluate_to_tensorflow(
    comp: federated_language.framework.ComputationBuildingBlock,
    bindings: dict[str, object],
) -> object:
  """Evaluates `comp` within a TensorFlow context, returning a tensor structure.

  Args:
    comp: A building block to evaluate. In order to evaluate to TensorFlow, this
      block must not contain any `Intrinsic`, `Data`, or `Placement` blocks, and
      must not contain `CompiledComputation` blocks of kinds other than
      `tensorflow`. `comp` must also have unique names.
    bindings: A mapping from names to values. Since names in `comp` must be
      unique, all block locals and lambda arguments can be inserted into this
      flat-level map.

  Returns:
    A structure of TensorFlow values representing the result of evaluating
    `comp`. Functional building blocks are represented as callables.

  Raises:
    XlaToTensorFlowError: If `comp` contains a `CompiledComputation` containing
      XLA.
    ExternalBlockToTensorFlowError: If `comp` contains an `Intrinsic`, `Data`,
      or `Placement` block.
    ValueError: If `comp` contains `CompiledCompilations` other than
      TensorFlow or XLA.
  """
  if isinstance(comp, federated_language.framework.Block):
    for name, value in comp.locals:
      bindings[name] = _evaluate_to_tensorflow(value, bindings)
    return _evaluate_to_tensorflow(comp.result, bindings)
  if isinstance(comp, federated_language.framework.CompiledComputation):
    kind = comp.to_proto().WhichOneof('computation')
    if kind == 'tensorflow':

      def call_concrete(*args):
        concrete = federated_language.framework.ConcreteComputation(
            computation_proto=comp.to_proto(),
            context_stack=federated_language.framework.get_context_stack(),
        )
        result = concrete(*args)
        if isinstance(
            comp.type_signature.result, federated_language.StructType
        ):
          return structure.from_container(result, recursive=True)
        return result

      return call_concrete
    if kind == 'xla':
      raise XlaToTensorFlowError(
          f'Cannot compile XLA subcomputation to TensorFlow:\n{comp}'
      )
    raise ValueError(f'Unexpected compiled computation kind:\n{kind}')
  if isinstance(comp, federated_language.framework.Call):
    function = _evaluate_to_tensorflow(comp.function, bindings)
    if comp.argument is None:
      return function()
    else:
      return function(_evaluate_to_tensorflow(comp.argument, bindings))
  if isinstance(comp, federated_language.framework.Lambda):
    if comp.parameter_type is None:
      return lambda: _evaluate_to_tensorflow(comp.result, bindings)
    else:

      def lambda_function(arg):
        bindings[comp.parameter_name] = arg
        return _evaluate_to_tensorflow(comp.result, bindings)

      return lambda_function
  if isinstance(comp, federated_language.framework.Reference):
    return bindings[comp.name]
  if isinstance(comp, federated_language.framework.Selection):
    return _evaluate_to_tensorflow(comp.source, bindings)[comp.as_index()]
  if isinstance(comp, federated_language.framework.Struct):
    elements = []
    for name, element in structure.iter_elements(comp):
      elements.append((name, _evaluate_to_tensorflow(element, bindings)))
    return structure.Struct(elements)
  if isinstance(comp, federated_language.framework.Literal):
    return tf.constant(comp.value)
  if isinstance(
      comp,
      (
          federated_language.framework.Intrinsic,
          federated_language.framework.Data,
          federated_language.framework.Placement,
      ),
  ):
    raise ExternalBlockToTensorFlowError(
        'Cannot evaluate intrinsic, data, or placement blocks to tensorflow,'
        f' found {comp}'
    )


def compile_local_computation_to_tensorflow(
    comp: federated_language.framework.ComputationBuildingBlock,
) -> federated_language.framework.ComputationBuildingBlock:
  """Compiles a fully specified local computation to TensorFlow.

  Args:
    comp: A `federated_language.framework.ComputationBuildingBlock` which can be
      compiled to TensorFlow. In order to compile a computation to TensorFlow,
      it must not contain 1. References to values defined outside of comp, 2.
      `Data`, `Intrinsic`, or `Placement` blocks, or 3. Calls to intrinsics or
      non-TensorFlow computations.

  Returns:
    A `federated_language.framework.ComputationBuildingBlock` containing a
    TensorFlow-only
    representation of `comp`. If `comp` is of functional type, this will be
    a `federated_language.framework.CompiledComputation`. Otherwise, it will be
    a
    `federated_language.framework.Call` which wraps a
    `federated_language.framework.CompiledComputation`.
  """
  if not isinstance(comp.type_signature, federated_language.FunctionType):
    lambda_wrapped = federated_language.framework.Lambda(None, None, comp)
    return federated_language.framework.Call(
        compile_local_computation_to_tensorflow(lambda_wrapped), None
    )

  parameter_type = comp.type_signature.parameter  # pytype: disable=attribute-error
  tensorflow_types.check_tensorflow_compatible_type(parameter_type)
  tensorflow_types.check_tensorflow_compatible_type(
      comp.type_signature.result  # pytype: disable=attribute-error
  )

  if (
      isinstance(comp, federated_language.framework.CompiledComputation)
      and comp.to_proto().WhichOneof('computation') == 'tensorflow'
  ):
    return comp

  # Ensure that unused values are removed and that reference bindings have
  # unique names.
  comp = unpack_compiled_computations(comp)
  comp = transformations.to_call_dominant(comp)

  if parameter_type is None:
    to_evaluate = federated_language.framework.Call(comp)

    @tensorflow_computation.tf_computation
    def result_computation():
      return _evaluate_to_tensorflow(to_evaluate, {})

  else:
    name_generator = federated_language.framework.unique_name_generator(comp)
    parameter_name = next(name_generator)
    to_evaluate = federated_language.framework.Call(
        comp,
        federated_language.framework.Reference(parameter_name, parameter_type),
    )

    @tensorflow_computation.tf_computation(parameter_type)
    def result_computation(arg):
      if isinstance(parameter_type, federated_language.StructType):
        arg = structure.from_container(arg, recursive=True)
      return _evaluate_to_tensorflow(to_evaluate, {parameter_name: arg})

  return result_computation.to_compiled_building_block()


def compile_local_subcomputations_to_tensorflow(
    comp: federated_language.framework.ComputationBuildingBlock,
) -> federated_language.framework.ComputationBuildingBlock:
  """Compiles subcomputations to TensorFlow where possible."""
  comp = unpack_compiled_computations(comp)
  local_cache = {}

  def _is_local(comp):
    cached = local_cache.get(comp, None)
    if cached is not None:
      return cached
    if isinstance(
        comp,
        (
            federated_language.framework.Intrinsic,
            federated_language.framework.Data,
            federated_language.framework.Placement,
        ),
    ) or federated_language.framework.contains_federated_types(
        comp.type_signature
    ):
      local_cache[comp] = False
      return False
    if (
        isinstance(comp, federated_language.framework.CompiledComputation)
        and comp.to_proto().WhichOneof('computation') == 'xla'
    ):
      local_cache[comp] = False
      return False
    for child in comp.children():
      if not _is_local(child):
        local_cache[comp] = False
        return False
    return True

  unbound_ref_map = federated_language.framework.get_map_of_unbound_references(
      comp
  )

  def _compile_if_local(comp):
    if _is_local(comp) and not unbound_ref_map[comp]:
      return compile_local_computation_to_tensorflow(comp), True
    return comp, False

  # Note: this transformation is preorder so that local subcomputations are not
  # first transformed to TensorFlow if they have a parent local computation
  # which could have instead been transformed into a larger single block of
  # TensorFlow.
  comp, _ = federated_language.framework.transform_preorder(
      comp, _compile_if_local
  )
  return comp


def parse_tff_to_tf(comp, grappler_config_proto):
  """Parses TFF construct `comp` into TensorFlow construct.

  Does not change the type signature of `comp`. Therefore may return either
  a `federated_language.framework.CompiledComputation` or a
  `federated_language.framework.Call` with no
  argument and function `federated_language.framework.CompiledComputation`.

  Args:
    comp: Instance of `federated_language.framework.ComputationBuildingBlock` to
      parse down to a single TF block.
    grappler_config_proto: An instance of `tf.compat.v1.ConfigProto` to
      configure Grappler graph optimization of the generated TensorFlow graph.
      If `grappler_config_proto` has
      `graph_options.rewrite_options.disable_meta_optimizer=True`, Grappler is
      bypassed.

  Returns:
    The result of parsing TFF to TF. If successful, this is either a single
    `federated_language.framework.CompiledComputation`, or a call to one. If
    unsuccessful,
    there may be more TFF constructs still remaining. Notice it is not the job
    of this function, but rather its callers, to check that the result of this
    parse is as expected.
  """
  tf_parsed = compile_local_computation_to_tensorflow(comp)

  # TODO: b/184883078 - Remove this check and trust Grappler to disable itself
  # based on the `disable_meta_optimizer` config.
  should_skip_grappler = (
      grappler_config_proto.HasField('graph_options')
      and grappler_config_proto.graph_options.HasField('rewrite_options')
      and grappler_config_proto.graph_options.rewrite_options.disable_meta_optimizer
  )
  if not should_skip_grappler:
    logging.info('Using Grappler on `MapReduceForm` TensorFlow graphs.')
    tf_parsed, _ = (
        compiled_computation_transformations.optimize_tensorflow_graphs(
            tf_parsed, grappler_config_proto
        )
    )

  return tf_parsed


def concatenate_function_outputs(first_function, second_function):
  """Constructs a new function concatenating the outputs of its arguments.

  Assumes that `first_function` and `second_function` already have unique
  names, and have declared parameters of the same type. The constructed
  function will bind its parameter to each of the parameters of
  `first_function` and `second_function`, and return the result of executing
  these functions in parallel and concatenating the outputs in a tuple.

  Args:
    first_function: Instance of `federated_language.framework.Lambda` whose
      result we wish to concatenate with the result of `second_function`.
    second_function: Instance of `federated_language.framework.Lambda` whose
      result we wish to concatenate with the result of `first_function`.

  Returns:
    A new instance of `federated_language.framework.Lambda` with unique names
    representing
    the computation described above.

  Raises:
    TypeError: If the arguments are not instances of
    `federated_language.framework.Lambda`,
    or declare parameters of different types.
  """

  py_typecheck.check_type(first_function, federated_language.framework.Lambda)
  py_typecheck.check_type(second_function, federated_language.framework.Lambda)
  federated_language.framework.check_has_unique_names(first_function)
  federated_language.framework.check_has_unique_names(second_function)

  if first_function.parameter_type != second_function.parameter_type:
    raise TypeError(
        'Must pass two functions which declare the same parameter '
        'type to `concatenate_function_outputs`; you have passed '
        'one function which declared a parameter of type {}, and '
        'another which declares a parameter of type {}'.format(
            first_function.type_signature, second_function.type_signature
        )
    )

  def _rename_first_function_arg(comp):
    if (
        isinstance(comp, federated_language.framework.Reference)
        and comp.name == first_function.parameter_name
    ):
      if comp.type_signature != second_function.parameter_type:
        raise AssertionError(
            '{}, {}'.format(comp.type_signature, second_function.parameter_type)
        )
      return (
          federated_language.framework.Reference(
              second_function.parameter_name, comp.type_signature
          ),
          True,
      )
    return comp, False

  first_function, _ = federated_language.framework.transform_postorder(
      first_function, _rename_first_function_arg
  )

  concatenated_function = federated_language.framework.Lambda(
      second_function.parameter_name,
      second_function.parameter_type,
      federated_language.framework.Struct(
          [first_function.result, second_function.result]
      ),
  )

  renamed, _ = tree_transformations.uniquify_reference_names(
      concatenated_function
  )

  return renamed
