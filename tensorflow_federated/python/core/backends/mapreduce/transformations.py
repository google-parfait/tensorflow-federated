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
from typing import Dict, List, Set, Tuple, Union

from absl import logging
import attr

from tensorflow_federated.python.common_libs import py_typecheck
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
  # Drop any unused subcomputations which may reference placements different
  # from the result.
  simplified, _ = transformations.transform_to_call_dominant(comp)
  unplaced, _ = tree_transformations.strip_placement(simplified)
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
      return_type = computation_types.FederatedType(
          computation_types.StructType([
              (None, call.type_signature.member) for call in calls
          ]),
          placement=calls[0].type_signature.placement)
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

  Under these conditions, this function will return two
  `building_blocks.Lambda`s `before` and `after` such that `comp` is
  semantically equivalent to the following expression*:

  ```
  (arg -> (let
    x=before(arg),
    y=intrinsic1(x[0]),
    z=intrinsic2(x[1]),
    ...
   in after(<arg, <y,z,...>>)))
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
  comp, _ = transformations.transform_to_call_dominant(comp)

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
  after_param_type = computation_types.StructType([
      ('original_arg', comp.parameter_type),
      ('intrinsic_results',
       computation_types.StructType([(f'{merged.uri}_result',
                                      merged.return_type)
                                     for merged in merged_intrinsics])),
  ])
  after_param_ref = building_blocks.Reference(after_param_name,
                                              after_param_type)
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
          [(comp.parameter_name,
            building_blocks.Selection(after_param_ref, name='original_arg'))] +
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


Index = Union[str, int]
Path = Union[Index, Tuple[Index, ...]]


def select_output_from_lambda(
    comp: building_blocks.Lambda,
    paths: Union[Path, List[Path]]) -> building_blocks.Lambda:
  """Constructs a new function with result of selecting `paths` from `comp`.

  Args:
    comp: Lambda computation with result type `tff.StructType` from which we
      wish to select the sub-results at `paths`.
    paths: Either a `Path` or list of `Path`s specifying the indices we wish to
      select from the result of `comp`. Each path must be a `tuple` of `str` or
      `int` indices from which to select an output. If `paths` is a list, the
      returned computation will have a `tff.StructType` result holding each of
      the specified selections.

  Returns:
    A transformed version of `comp` with result value the selection from the
    result of `comp` specified by `paths`.
  """
  comp.check_lambda()
  comp.type_signature.result.check_struct()

  def _select_path(result, path: Path):
    if not isinstance(path, tuple):
      path = (path,)
    for index in path:
      if result.is_struct():
        result = result[index]
      elif isinstance(index, str):
        result = building_blocks.Selection(result, name=index)
      elif isinstance(index, int):
        result = building_blocks.Selection(result, index=index)
      else:
        raise TypeError('Invalid selection type: expected `str` or `int`, '
                        f'found value `{index}` of type `{type(index)}`.')
    return result

  if isinstance(paths, list):
    elements = [_select_path(comp.result, path) for path in paths]
    result = building_blocks.Struct(elements)
  else:
    result = _select_path(comp.result, paths)
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
