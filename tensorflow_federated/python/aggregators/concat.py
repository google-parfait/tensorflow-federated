# Copyright 2021, Google LLC.
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
"""Factory for concatenation of input to a single tensor."""

import functools
from typing import TypeVar

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


_T = TypeVar('_T', bound=factory.AggregationFactory)


def _concat_impl(struct):
  """Flattens each tensor in the structure and concats them into a vector."""
  flattened_vectors = [tf.reshape(x, [-1]) for x in tf.nest.flatten(struct)]
  return tf.concat(flattened_vectors, axis=0)


def _unconcat_impl(concatenated_tensor, original_structure):
  """Applies the inverse of `_concat_impl` given the original structure."""
  start_location, split_tensors = 0, []
  for original_tensor in tf.nest.flatten(original_structure):
    length = int(functools.reduce(lambda a, b: a * b, original_tensor.shape, 1))
    split_vector = concatenated_tensor[start_location : start_location + length]
    split_tensors.append(tf.reshape(split_vector, original_tensor.shape))
    start_location += length
  return tf.nest.pack_sequence_as(original_structure, split_tensors)


def _next_fn_impl(
    state, value, concat_fn, unconcat_fn, inner_agg_process, weight=None
):
  """Implements the next_fn for concat_factory's resulting AggregationProcess."""
  concat_value = intrinsics.federated_map(concat_fn, value)
  if weight is None:
    inner_agg_output = inner_agg_process.next(state, concat_value)
  else:
    inner_agg_output = inner_agg_process.next(state, concat_value, weight)

  unconcat_value = intrinsics.federated_map(
      unconcat_fn, inner_agg_output.result
  )
  return measured_process.MeasuredProcessOutput(
      state=inner_agg_output.state,
      result=unconcat_value,
      measurements=inner_agg_output.measurements,
  )


def create_concat_fns(
    value_type: factory.ValueType,
) -> tuple[computation_base.Computation, computation_base.Computation]:
  """Creates the forward and backward flattening/concatenation functions."""
  # As the factory alters the tensor specs, we compute the Python structure
  # of the types for the unconcat procedure.
  if isinstance(
      value_type, computation_types.StructWithPythonType
  ) and type_analysis.is_structure_of_tensors(value_type):
    original_structure = type_conversions.structure_from_tensor_type_tree(
        lambda x: tf.TensorSpec(x.shape, x.dtype), value_type
    )
  elif isinstance(value_type, computation_types.TensorType):
    original_structure = tf.TensorSpec(value_type.shape, value_type.dtype)
  else:
    raise TypeError(
        'Expected `value_type` to be `TensorType` or '
        '`StructWithPythonType` containing only `TensorType`. '
        f'Found type: {repr(value_type)}'
    )

  _check_component_dtypes(value_type)

  @tensorflow_computation.tf_computation(value_type)
  def concat(struct):
    return _concat_impl(struct)

  @tensorflow_computation.tf_computation(concat.type_signature.result)
  def unconcat(concatenated_tensor):
    return _unconcat_impl(concatenated_tensor, original_structure)

  return concat, unconcat


def _check_component_dtypes(value_type):
  """Checks the component tensor dtypes of the input `value_type`."""
  component_dtypes = set([v.dtype for v in structure.flatten(value_type)])
  # Check that all component tensors have the same dtype.
  if len(component_dtypes) != 1:
    raise TypeError(
        'Component tensors of the structure should have the same '
        f'dtype. Found dtypes: {component_dtypes}.'
    )

  # Restrict dtypes to integers and floats for now.
  if not (
      type_analysis.is_structure_of_floats(value_type)
      or type_analysis.is_structure_of_integers(value_type)
  ):
    raise TypeError(
        'Components of `value_type` must all be integers or '
        f'floats. Found {value_type}.'
    )


def _unweighted_concat_factory(inner_agg_factory):
  """Creates a unweighted factory for flattening and concatenation."""

  class UnweightedConcatFactory(factory.UnweightedAggregationFactory):
    """A concat_factory with type `UnweightedAggregationFactory`."""

    def create(self, value_type) -> aggregation_process.AggregationProcess:
      concat_fn, unconcat_fn = create_concat_fns(value_type)
      inner_agg_process = inner_agg_factory.create(
          concat_fn.type_signature.result
      )
      init_fn = inner_agg_process.initialize
      state_type = init_fn.type_signature.result

      @federated_computation.federated_computation(
          state_type,
          computation_types.FederatedType(value_type, placements.CLIENTS),
      )
      def next_fn(state, value):
        return _next_fn_impl(
            state, value, concat_fn, unconcat_fn, inner_agg_process
        )

      return aggregation_process.AggregationProcess(init_fn, next_fn)

  return UnweightedConcatFactory()


def _weighted_concat_factory(inner_agg_factory):
  """Creates a weighted factory for flattening and concatenation."""

  class WeightedConcatFactory(factory.WeightedAggregationFactory):
    """A concat_factory with type `WeightedAggregationFactory`."""

    def create(
        self, value_type, weight_type
    ) -> aggregation_process.AggregationProcess:
      concat_fn, unconcat_fn = create_concat_fns(value_type)
      inner_agg_process = inner_agg_factory.create(
          concat_fn.type_signature.result, weight_type
      )
      init_fn = inner_agg_process.initialize

      @federated_computation.federated_computation(
          init_fn.type_signature.result,
          computation_types.FederatedType(value_type, placements.CLIENTS),
          computation_types.FederatedType(weight_type, placements.CLIENTS),
      )
      def next_fn(state, value, weight):
        return _next_fn_impl(
            state, value, concat_fn, unconcat_fn, inner_agg_process, weight
        )

      return aggregation_process.AggregationProcess(init_fn, next_fn)

  return WeightedConcatFactory()


def concat_factory(inner_agg_factory: _T) -> _T:
  """Aggregation factory for concatenation of input to a single tensor.

  The created `tff.templates.AggregationProcess` takes the input structure,
  reshapes each tensor in the structure to a rank-1 tensor, and concatenates
  them into a single tensor, which is passed to the inner aggregation. After the
  inner aggregation, the concatenated tensor is split and packed into the
  original structure.

  For example, if this factory receives TFF type `<float32[N],float32[P,Q]>`,
  then the `inner_agg_factory` will operate on `<float32[N + P * Q]>`. Note that
  the factory expects all tensors in the structure to have the same numeric
  dtype; for example, an input `value_type` of `<int32[N],float32[P,Q]>` or
  `<string[N]>` will raise an error.

  This aggregator may be useful as a preprocessing step for algorithms that need
  to operate on client values as a single vector; for example, the algorithm may
  need to apply randomized Hadamard transform with zero padding on the client
  vectors, in which case applying the transform separately on each component
  may not be identical to applying the transform to the single vector at once.

  The returned factory takes its weightedness
  (`UnweightedAggregationFactory` vs. `WeightedAggregationFactory`) from the
  `inner_agg_factory`.

  This factory only accepts `value_type` of either `tff.TensorType` or
  `tff.StructWithPythonType` and expects the dtype of component tensors to be
  either all real integers or all real floats.

  Args:
    inner_agg_factory: A factory specifying the type of aggregation to be done
      after flattening and concatenating the structure into a single vector.

  Returns:
    An aggregation factory that flattens and concatenates the components of a
    tensor structure into a single rank-1 tensor.
  """
  if isinstance(inner_agg_factory, factory.WeightedAggregationFactory):
    return _weighted_concat_factory(inner_agg_factory)
  else:
    return _unweighted_concat_factory(inner_agg_factory)
