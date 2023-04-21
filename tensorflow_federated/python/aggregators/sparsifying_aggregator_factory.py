# Copyright 2023, The TensorFlow Federated Authors.
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
"""Libraries for more preformant aggregation with lots of zeros."""

import collections
import functools
import operator
from typing import Any, Union

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


@federated_computation.federated_computation
def _empty_state_initialize():
  empty_state = ()
  return intrinsics.federated_value(empty_state, placements.SERVER)


def _is_sparse_tensor_structure(
    value_type_spec: computation_types.Type,
) -> bool:
  return (
      isinstance(value_type_spec, computation_types.StructWithPythonType)
      and value_type_spec.python_container is tf.SparseTensor
  )


def _build_sparsify_computation(threshold: int) -> computation_base.Computation:
  """Builds a tff.Computation that will sparsify a tensor value depending on size."""

  @tensorflow_computation.tf_computation
  def _sparsify(values):
    """Convert a dense tf.Tensor to a tf.SparseTensor by dropping zeros."""

    def _maybe_sparsify(value: tf.Tensor) -> Union[tf.Tensor, tf.SparseTensor]:
      if not value.shape.is_fully_defined():
        raise ValueError(
            'Cannot apply sparsification to tensors that do not '
            f'have fully defined static shapes. Got {value.shape}'
        )
      # Don't bother sparsifying small tensors.
      if functools.reduce(operator.mul, value.shape.as_list(), 1) < threshold:
        return value
      # NOTE: `from_dense` drops all the zeros values.
      return tf.sparse.from_dense(value)

    return tf.nest.map_structure(_maybe_sparsify, values)

  return _sparsify


def _build_sparse_zero(
    type_spec: computation_types.StructWithPythonType,
    dense_shape: tf.TensorShape,
) -> Union[tf.SparseTensor, tf.Tensor]:
  num_dense_dimensions = sum(type_spec.dense_shape.shape.as_list())
  return tf.SparseTensor(
      indices=tf.zeros(shape=[0, num_dense_dimensions], dtype=tf.int64),
      values=tf.constant([], dtype=type_spec.values.dtype),
      dense_shape=tf.constant(dense_shape.as_list(), dtype=tf.int64))


def _build_value_zeros(
    value_type_spec: Union[
        computation_types.TensorType, computation_types.StructType
    ],
    dense_shapes: Union[structure.Struct, tf.TensorShape],
) -> Any:
  """Construct a potentially sparse `zero` value for a computation_types.Type."""
  if value_type_spec.is_tensor():
    return tf.zeros(shape=value_type_spec.shape, dtype=value_type_spec.dtype)
  elif _is_sparse_tensor_structure(value_type_spec):
    return _build_sparse_zero(value_type_spec, dense_shapes)
  elif value_type_spec.is_struct():
    zipped_names_types_shapes = zip(
        structure.iter_elements(value_type_spec), dense_shapes
    )
    child_zeros = structure.Struct(
        [
            (name, _build_value_zeros(child_type, dense_shape))
            for (name, child_type), dense_shape in zipped_names_types_shapes
        ]
    )
    try:
      return type_conversions.type_to_py_container(child_zeros, value_type_spec)
    except TypeError as e:
      raise TypeError(
          f'Cannot build zeros for type: {value_type_spec!r}') from e
  else:
    raise TypeError(f'Cannot build zeros for type: {value_type_spec!r}')


class SparsifyingSumFactory(factory.UnweightedAggregationFactory):
  """An UnweightedAggregationFactory that sums sparse tensors.

  This factory produces aggregation processes that convert dense `tf.Tensor`s to
  `tf.SparseTensor`s before aggregation by dropping zero parameters, summing in
  the sparse space, and finally converting back to `tf.Tensor` after summation.

  The AggregationProcess created will *not* sparsify tensors with fewer than
  `element_threhsold` parameters.

  While communication protocol compression may already prevents large runs of
  zeros from being sent over the wire inefficiently (negating a need for
  sparsity for communication), this aggregator could additionally reduce the
  amount of compute needed. Particularly in naive implementations of learning
  algorithms, _all_ parameters, regardless of their value, are aggregated,
  potentially using a large amount of FLOPs for large models. This aggregator
  can provide some compute savings when a large number of coordinates will be
  zero, which does not need to be communicated for summation.
  """

  def __init__(self, element_threshold: int = 100):
    """Constructs a `SparsifyingSumFactory`.

    Args:
      element_threshold: A non-negative integer minimum size (in number of
        parameters) before a tensor is sparisifed before aggregation.
    """
    if element_threshold < 0:
      raise ValueError(
          f'element_threshold must be non-negative, got {element_threshold}'
      )
    self._element_threshold = element_threshold

  def create(
      self,
      value_type: Union[
          computation_types.TensorType, computation_types.StructType
      ],
  ) -> aggregation_process.AggregationProcess:
    @federated_computation.federated_computation(
        computation_types.at_server(()),
        computation_types.at_clients(value_type),
    )
    def next_fn(state, client_values):
      del state  # Unused.
      sparse_client_values = intrinsics.federated_map(
          _build_sparsify_computation(self._element_threshold), client_values
      )
      dense_shapes = type_conversions.structure_from_tensor_type_tree(
          lambda t: t.shape, client_values.type_signature.member
      )
      if client_values.type_signature.member.is_struct():
        dense_shape_structure = structure.from_container(
            dense_shapes, recursive=True
        )
      else:
        dense_shape_structure = dense_shapes
      # The type_signature of the sparse_client_values has the _shape_ of the
      # `dense_shape` tensor for SparseTensors, but we need the _value_ of the
      # `dense_shape` tensor when constructing the zeros. This is available from
      # the type_signature of the _dense_ client tensors coming in (extracted
      # above) so we also pass in the `dense_shapes` structure with this
      # information.
      sparse_client_values_type = sparse_client_values.type_signature.member

      @tensorflow_computation.tf_computation
      def zero_values():
        return _build_value_zeros(sparse_client_values_type,
                                  dense_shape_structure)

      zero_values = zero_values()
      client_values_type = client_values.type_signature.member

      @tensorflow_computation.tf_computation
      def build_count_zeros():
        if client_values_type.is_struct():
          return type_conversions.structure_from_tensor_type_tree(
              lambda *_: tf.zeros(dtype=tf.int64, shape=[]),
              client_values_type,
          )
        elif client_values_type.is_tensor():
          return tf.zeros(dtype=tf.int64, shape=[])
        else:
          raise TypeError('Internal type error, code is incorrect.')

      zero_counts = build_count_zeros()

      def replace_zero_dimensions_with_none(
          type_spec: computation_types.Type,
      ) -> computation_types.Type:
        if type_spec.is_tensor():
          return computation_types.TensorType(
              dtype=type_spec.dtype,
              shape=[None if dim == 0 else dim for dim in type_spec.shape],
          )
        elif type_spec.is_struct():
          elements = [
              (name, replace_zero_dimensions_with_none(element))
              for name, element in structure.iter_elements(type_spec)
          ]
          if isinstance(type_spec, computation_types.StructWithPythonType):
            return computation_types.StructWithPythonType(
                elements, type_spec.python_container
            )
          return computation_types.StructType(elements)
        else:
          # This indicates a coding error rather than an invalid input, a recent
          # code modification must have broke some assumption.
          raise TypeError('internal error')

      # Replace any `0` dimension in the `tf.SparseTensor` used for
      # accumulation, such as the `.values` tensor. This will allow us
      # to "grow" that dimension with new values as they are summed.
      growing_sparse_tensor_type_spec = replace_zero_dimensions_with_none(
          zero_values.type_signature
      )

      @tensorflow_computation.tf_computation(
          (growing_sparse_tensor_type_spec, zero_counts.type_signature),
          growing_sparse_tensor_type_spec,
      )
      def _sparse_add(partial_sum_state, summand_values):
        partial_sum_values, partial_coordinate_counts = partial_sum_state

        def _sparse_or_dense_add(a, b):
          if isinstance(a, tf.SparseTensor):
            return tf.sparse.add(a, b)
          else:
            return tf.add(a, b)

        try:
          values_sums = tf.nest.map_structure(
              _sparse_or_dense_add, partial_sum_values, summand_values
          )
        except ValueError as e:
          raise ValueError(f'{partial_sum_values!r}\n{summand_values!r}') from e

        # Count how many coordinates were in the sparse tensors and output
        # as measurements to give modellers a sense of how the process is
        # performing.
        def accumulate_coordinate_count_if_sparse(partial_count, client_value):
          if isinstance(client_value, tf.SparseTensor):
            return partial_count + tf.size(
                client_value.values, out_type=tf.int64)
          else:
            return partial_count

        coordinate_count_sums = tf.nest.map_structure(
            accumulate_coordinate_count_if_sparse,
            partial_coordinate_counts,
            summand_values,
        )
        return values_sums, coordinate_count_sums

      @tensorflow_computation.tf_computation
      def _sparse_merge(a, b):
        a_values, a_counts = a
        b_values, b_counts = b

        def _sparse_or_dense_add(a, b):
          if isinstance(a, tf.SparseTensor):
            return tf.sparse.add(a, b)
          else:
            return tf.add(a, b)

        values_sums = tf.nest.map_structure(
            _sparse_or_dense_add, a_values, b_values
        )
        counts_sums = tf.nest.map_structure(tf.add, a_counts, b_counts)
        return values_sums, counts_sums

      @tensorflow_computation.tf_computation
      def _densify(sums):
        sparse_values, client_counts = sums

        # Count how many coordinates are in the summation. This can be
        # interesting compare against `client_counts` to determine if the
        # clients heavily overlap or not (e.g. aggregate_counts == client_counts
        # indicates mostly non-overlap, aggregat_counts * num_clients ==
        # client_counts would indicate total overlap in sparse updates).
        def count_coordinates_if_sparse(tensor):
          if isinstance(tensor, tf.SparseTensor):
            return tf.size(tensor.values, out_type=tf.int64)
          else:
            return tf.constant(0, dtype=tf.int64)

        aggregate_counts = tf.nest.map_structure(count_coordinates_if_sparse,
                                                 sparse_values)

        def _densify_if_sparse(tensor, shape):
          if isinstance(tensor, tf.SparseTensor):
            # We add an additional reshape() call here to get a fully-defined
            # tensor out of this function. Otherwise the federated_aggregate
            # ends up with a known rank, but unknown dimensions sparse tensor
            # which propagates here.
            return tf.reshape(tf.sparse.to_dense(tensor), shape=shape)
          return tensor

        return (
            tf.nest.map_structure(
                _densify_if_sparse, sparse_values, dense_shapes
            ),
            client_counts,
            aggregate_counts,
        )

      composite_zero = (zero_values, zero_counts)
      dense_sums, client_coordinate_counts, aggregate_coordinate_counts = (
          intrinsics.federated_aggregate(
              value=sparse_client_values,
              zero=composite_zero,
              accumulate=_sparse_add,
              merge=_sparse_merge,
              report=_densify,
          )
      )
      empty_state = intrinsics.federated_value((), placements.SERVER)
      measurements = intrinsics.federated_zip(
          collections.OrderedDict(
              client_coordinate_counts=client_coordinate_counts,
              aggregate_coordinate_counts=aggregate_coordinate_counts,
          )
      )
      return measured_process.MeasuredProcessOutput(
          empty_state, dense_sums, measurements
      )

    return aggregation_process.AggregationProcess(
        initialize_fn=_empty_state_initialize, next_fn=next_fn
    )
