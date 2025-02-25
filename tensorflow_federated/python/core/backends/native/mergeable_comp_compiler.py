# Copyright 2021, The TensorFlow Federated Authors.
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
"""A MergeableCompForm compiler for the native backend."""

import federated_language
from tensorflow_federated.python.core.backends.mapreduce import compiler
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_building_block_factory
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_computation_factory
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_tree_transformations
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.execution_contexts import mergeable_comp_execution_context


def _compile_to_tf(fn):
  simplified = transformations.to_call_dominant(fn)
  unplaced, _ = tree_transformations.strip_placement(simplified)
  return compiler.compile_local_subcomputations_to_tensorflow(unplaced)


def _select_output_result_and_wrap_as_noarg_tensorflow(
    fn: federated_language.framework.Lambda,
    path: federated_language.framework.Path,
) -> federated_language.framework.ConcreteComputation:
  selected_and_wrapped = federated_language.framework.Lambda(
      None,
      None,
      federated_language.framework.select_output_from_lambda(fn, path).result,
  )
  selected_and_compiled = _compile_to_tf(selected_and_wrapped)
  return federated_language.framework.ConcreteComputation(
      computation_proto=selected_and_compiled.to_proto(),
      context_stack=federated_language.framework.get_context_stack(),
  )


def _select_output_result_and_wrap_as_tensorflow(
    fn: federated_language.framework.Lambda,
    path: federated_language.framework.Path,
) -> federated_language.framework.ConcreteComputation:
  selected_fn = federated_language.framework.select_output_from_lambda(
      fn, path
  ).result
  selected_and_compiled = _compile_to_tf(selected_fn)
  return federated_language.framework.ConcreteComputation(
      computation_proto=selected_and_compiled.to_proto(),
      context_stack=federated_language.framework.get_context_stack(),
  )


def _extract_federated_aggregate_computations(
    before_agg: federated_language.framework.Lambda,
):
  """Extracts aggregate computations from `before_agg`.

  Args:
    before_agg: a `federated_language.framework.ComputationBuildingBlock`
      representing the before-aggregate portion of a computation split on
      `federated_aggregate`.

  Returns:
    A tuple of four ConcreteComputations corresponding to the aggregate
    functions in `before_agg`.
  """
  federated_aggregate_arguments = (
      federated_language.framework.select_output_from_lambda(
          before_agg, 'federated_aggregate_param'
      )
  )
  # Index 0 is the value to be aggregated, so we skip.
  # We compile the federated aggregate functions to Tensorflow since simply
  # isolating them from their defining scope can leave unused references in the
  # bodies of these computations which are undefined in the smaller sub-scope.
  # Compilation to TF ensures these references are removed.
  zero = _select_output_result_and_wrap_as_noarg_tensorflow(
      federated_aggregate_arguments, 1
  )
  accumulate = _select_output_result_and_wrap_as_tensorflow(
      federated_aggregate_arguments, 2
  )
  merge = _select_output_result_and_wrap_as_tensorflow(
      federated_aggregate_arguments, 3
  )
  report = _select_output_result_and_wrap_as_tensorflow(
      federated_aggregate_arguments, 4
  )
  return zero, accumulate, merge, report


def _ensure_lambda(
    building_block: federated_language.framework.ComputationBuildingBlock,
) -> federated_language.framework.Lambda:
  """Wraps a functional building block as a lambda if necessary."""
  if not isinstance(
      building_block.type_signature, federated_language.FunctionType
  ):
    raise ValueError(
        'Expected a `federated_language.FunctionType`, found'
        f' {building_block.type_signature}.'
    )
  if not isinstance(building_block, federated_language.framework.Lambda):
    if building_block.type_signature.parameter is not None:  # pytype: disable=attribute-error
      name_generator = federated_language.framework.unique_name_generator(
          building_block
      )
      parameter_name = next(name_generator)
      argument = federated_language.framework.Reference(
          parameter_name,
          building_block.type_signature.parameter,  # pytype: disable=attribute-error
      )
      parameter_type = argument.type_signature
    else:
      argument = None
      parameter_type = None
      parameter_name = None
    result = federated_language.framework.Call(building_block, argument)
    building_block = federated_language.framework.Lambda(
        parameter_name, parameter_type, result
    )
  return building_block


def compile_to_mergeable_comp_form(
    comp: federated_language.framework.ConcreteComputation,
) -> mergeable_comp_execution_context.MergeableCompForm:
  """Compiles a computation with a single aggregation to `MergeableCompForm`.

  Compilation proceeds by splitting on the lone aggregation, and using the
  aggregation's internal functions to generate a semantically equivalent
  instance of `mergeable_comp_execution_context.MergeableCompForm`.

  Args:
    comp: Instance of `federated_language.framework.ConcreteComputation` to
      compile. Assumed to be representable as a computation with a single
      aggregation in its body, so that for example two parallel aggregations are
      allowed, but multiple dependent aggregations are disallowed. Additionally
      assumed to be of functional type.

  Returns:
    A semantically equivalent instance of
    `mergeable_comp_execution_context.MergeableCompForm`.

  Raises:
    TypeError: If `comp` is not a building block, or is not of functional TFF
      type.
    ValueError: If `comp` cannot be represented as a computation with at most
    one aggregation in its body.
  """
  original_return_type = comp.type_signature.result
  building_block = comp.to_building_block()
  lam = _ensure_lambda(building_block)
  lowered_bb, _ = (
      tensorflow_tree_transformations.replace_intrinsics_with_bodies(lam)
  )

  # We transform the body of this computation to easily preserve the top-level
  # lambda required by force-aligning.
  call_dominant_body_bb = transformations.to_call_dominant(lowered_bb.result)
  call_dominant_bb = federated_language.framework.Lambda(
      lowered_bb.parameter_name,
      lowered_bb.parameter_type,
      call_dominant_body_bb,
  )

  # This check should not throw false positives because we just ensured we are
  # in call-dominant form.
  federated_language.framework.check_aggregate_not_dependent_on_aggregate(
      call_dominant_bb
  )

  before_agg, after_agg = transformations.force_align_and_split_by_intrinsics(
      call_dominant_bb,
      [tensorflow_building_block_factory.create_null_federated_aggregate()],
  )

  # Construct a report function which accepts the result of merge.
  merge_fn_type = before_agg.type_signature.result['federated_aggregate_param'][
      3
  ]  # pytype: disable=unsupported-operands
  report_proto, report_type = tensorflow_computation_factory.create_identity(
      merge_fn_type.result
  )
  identity_report = federated_language.framework.CompiledComputation(
      report_proto, type_signature=report_type
  )

  zero_comp, accumulate_comp, merge_comp, report_comp = (
      _extract_federated_aggregate_computations(before_agg)
  )

  before_agg_callable = federated_language.framework.ConcreteComputation(
      computation_proto=before_agg.to_proto(),
      context_stack=federated_language.framework.get_context_stack(),
  )
  after_agg_callable = federated_language.framework.ConcreteComputation(
      computation_proto=after_agg.to_proto(),
      context_stack=federated_language.framework.get_context_stack(),
  )

  if before_agg.type_signature.parameter is not None:
    # TODO: b/147499373 - If None-arguments were uniformly represented as empty
    # tuples, we would be able to avoid this (and related) ugly casing.

    @federated_language.federated_computation(
        before_agg.type_signature.parameter
    )
    def up_to_merge_computation(arg):
      federated_aggregate_args = before_agg_callable(arg)[
          'federated_aggregate_param'
      ]
      value_to_aggregate = federated_aggregate_args[0]
      zero = zero_comp()
      return federated_language.federated_aggregate(
          value_to_aggregate, zero, accumulate_comp, merge_comp, identity_report
      )

    @federated_language.federated_computation(
        before_agg.type_signature.parameter,
        federated_language.FederatedType(
            identity_report.type_signature.result,  # pytype: disable=attribute-error
            federated_language.SERVER,
        ),
    )
    def after_merge_computation(top_level_arg, merge_result):
      reported_result = federated_language.federated_map(
          report_comp, merge_result
      )
      return after_agg_callable(top_level_arg, [reported_result])

  else:

    @federated_language.federated_computation()
    def up_to_merge_computation():
      federated_aggregate_args = before_agg_callable()[
          'federated_aggregate_param'
      ]
      value_to_aggregate = federated_aggregate_args[0]
      zero = zero_comp()
      return federated_language.federated_aggregate(
          value_to_aggregate, zero, accumulate_comp, merge_comp, identity_report
      )

    @federated_language.federated_computation(
        federated_language.FederatedType(
            identity_report.type_signature.result,  # pytype: disable=attribute-error
            federated_language.SERVER,
        )
    )
    def after_merge_computation(merge_result):
      reported_result = federated_language.federated_map(
          report_comp, merge_result
      )
      return after_agg_callable([[reported_result]])

  annotated_type_signature = federated_language.FunctionType(
      after_merge_computation.type_signature.parameter, original_return_type
  )
  # Note: Update to latest version of `federated-language` to use a public
  # property and remove the pylint directive.
  after_merge_computation = federated_language.framework.ConcreteComputation(
      computation_proto=after_merge_computation.to_proto(),
      context_stack=after_merge_computation._context_stack,  # pylint: disable=protected-access
      annotated_type=annotated_type_signature,
  )

  return mergeable_comp_execution_context.MergeableCompForm(
      up_to_merge=up_to_merge_computation,
      merge=merge_comp,
      after_merge=after_merge_computation,
  )
