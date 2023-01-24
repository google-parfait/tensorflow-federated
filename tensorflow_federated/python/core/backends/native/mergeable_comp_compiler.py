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

from tensorflow_federated.python.core.backends.mapreduce import compiler
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.execution_contexts import mergeable_comp_execution_context
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types


def _compile_to_tf(fn):
  simplified = transformations.to_call_dominant(fn)
  unplaced, _ = tree_transformations.strip_placement(simplified)
  return compiler.compile_local_subcomputations_to_tensorflow(unplaced)


def _select_output_result_and_wrap_as_noarg_tensorflow(
    fn: building_blocks.Lambda, path: building_block_factory.Path
) -> computation_impl.ConcreteComputation:
  selected_and_wrapped = building_blocks.Lambda(
      None,
      None,
      building_block_factory.select_output_from_lambda(fn, path).result,
  )
  selected_and_compiled = _compile_to_tf(selected_and_wrapped)
  return computation_impl.ConcreteComputation.from_building_block(
      selected_and_compiled
  )


def _select_output_result_and_wrap_as_tensorflow(
    fn: building_blocks.Lambda, path: building_block_factory.Path
) -> computation_impl.ConcreteComputation:
  selected_fn = building_block_factory.select_output_from_lambda(
      fn, path
  ).result
  selected_and_compiled = _compile_to_tf(selected_fn)
  return computation_impl.ConcreteComputation.from_building_block(
      selected_and_compiled
  )


def _extract_federated_aggregate_computations(
    before_agg: building_blocks.Lambda,
):
  """Extracts aggregate computations from `before_agg`.

  Args:
    before_agg: a `building_blocks.ComputationBuildingBlock` representing the
      before-aggregate portion of a computation split on `federated_aggregate`.

  Returns:
    A tuple of four ConcreteComputations corresponding to the aggregate
    functions in `before_agg`.
  """
  federated_aggregate_arguments = (
      building_block_factory.select_output_from_lambda(
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
    building_block: building_blocks.ComputationBuildingBlock,
) -> building_blocks.Lambda:
  """Wraps a functional building block as a lambda if necessary."""
  building_block.type_signature.check_function()
  if not isinstance(building_block, building_blocks.Lambda):
    if building_block.type_signature.parameter is not None:
      name_generator = building_block_factory.unique_name_generator(
          building_block
      )
      parameter_name = next(name_generator)
      argument = building_blocks.Reference(
          parameter_name, building_block.type_signature.parameter
      )
      parameter_type = argument.type_signature
    else:
      argument = None
      parameter_type = None
      parameter_name = None
    result = building_blocks.Call(building_block, argument)
    building_block = building_blocks.Lambda(
        parameter_name, parameter_type, result
    )
  return building_block


def compile_to_mergeable_comp_form(
    comp: computation_impl.ConcreteComputation,
) -> mergeable_comp_execution_context.MergeableCompForm:
  """Compiles a computation with a single aggregation to `MergeableCompForm`.

  Compilation proceeds by splitting on the lone aggregation, and using the
  aggregation's internal functions to generate a semantically equivalent
  instance of `mergeable_comp_execution_context.MergeableCompForm`.

  Args:
    comp: Instance of `computation_impl.ConcreteComputation` to compile. Assumed
      to be representable as a computation with a single aggregation in its
      body, so that for example two parallel aggregations are allowed, but
      multiple dependent aggregations are disallowed. Additionally assumed to be
      of functional type.

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
  lowered_bb, _ = tree_transformations.replace_intrinsics_with_bodies(lam)

  # We transform the body of this computation to easily preserve the top-level
  # lambda required by force-aligning.
  call_dominant_body_bb = transformations.to_call_dominant(lowered_bb.result)
  call_dominant_bb = building_blocks.Lambda(
      lowered_bb.parameter_name,
      lowered_bb.parameter_type,
      call_dominant_body_bb,
  )

  # This check should not throw false positives because we just ensured we are
  # in call-dominant form.
  tree_analysis.check_aggregate_not_dependent_on_aggregate(call_dominant_bb)

  before_agg, after_agg = transformations.force_align_and_split_by_intrinsics(
      call_dominant_bb,
      [building_block_factory.create_null_federated_aggregate()],
  )

  # Construct a report function which accepts the result of merge.
  merge_fn_type = before_agg.type_signature.result['federated_aggregate_param'][
      3
  ]
  identity_report = computation_impl.ConcreteComputation.from_building_block(
      building_block_factory.create_compiled_identity(merge_fn_type.result)
  )

  zero_comp, accumulate_comp, merge_comp, report_comp = (
      _extract_federated_aggregate_computations(before_agg)
  )

  before_agg_callable = (
      computation_impl.ConcreteComputation.from_building_block(before_agg)
  )
  after_agg_callable = computation_impl.ConcreteComputation.from_building_block(
      after_agg
  )

  if before_agg.type_signature.parameter is not None:
    # TODO(b/147499373): If None-arguments were uniformly represented as empty
    # tuples, we would be able to avoid this (and related) ugly casing.

    @federated_computation.federated_computation(
        before_agg.type_signature.parameter
    )
    def up_to_merge_computation(arg):
      federated_aggregate_args = before_agg_callable(arg)[
          'federated_aggregate_param'
      ]
      value_to_aggregate = federated_aggregate_args[0]
      zero = zero_comp()
      return intrinsics.federated_aggregate(
          value_to_aggregate, zero, accumulate_comp, merge_comp, identity_report
      )

    @federated_computation.federated_computation(
        before_agg.type_signature.parameter,
        computation_types.at_server(identity_report.type_signature.result),
    )
    def after_merge_computation(top_level_arg, merge_result):
      reported_result = intrinsics.federated_map(report_comp, merge_result)
      return after_agg_callable(top_level_arg, [reported_result])

  else:

    @federated_computation.federated_computation()
    def up_to_merge_computation():
      federated_aggregate_args = before_agg_callable()[
          'federated_aggregate_param'
      ]
      value_to_aggregate = federated_aggregate_args[0]
      zero = zero_comp()
      return intrinsics.federated_aggregate(
          value_to_aggregate, zero, accumulate_comp, merge_comp, identity_report
      )

    @federated_computation.federated_computation(
        computation_types.at_server(identity_report.type_signature.result)
    )
    def after_merge_computation(merge_result):
      reported_result = intrinsics.federated_map(report_comp, merge_result)
      return after_agg_callable([[reported_result]])

  annotated_type_signature = computation_types.FunctionType(
      after_merge_computation.type_signature.parameter, original_return_type
  )
  after_merge_computation = computation_impl.ConcreteComputation.with_type(
      after_merge_computation, annotated_type_signature
  )

  return mergeable_comp_execution_context.MergeableCompForm(
      up_to_merge=up_to_merge_computation,
      merge=merge_comp,
      after_merge=after_merge_computation,
  )
