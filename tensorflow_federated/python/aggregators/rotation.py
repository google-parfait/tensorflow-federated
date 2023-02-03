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
"""A tff.aggregator for flattening coordinate values across vector dimensions."""

import collections
import math
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import hadamard
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

SEED_TF_DTYPE = tf.int64
SEED_TFF_TYPE = computation_types.TensorType(SEED_TF_DTYPE, [2])
OUTPUT_TF_DTYPE = tf.float32


class HadamardTransformFactory(factory.UnweightedAggregationFactory):
  """`UnweightedAggregationFactory` for fast Walsh-Hadamard transform.

  The created `tff.templates.AggregationProcess` takes an input structure
  and applies the randomized fast Walsh-Hadamard transform to each tensor in the
  structure, reshaped to a rank-1 tensor in `O(d*log(d))` time, where `d` is the
  number of elements of the tensor.

  See https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform.

  Specifically, for each tensor, the following operations are first performed at
  `tff.CLIENTS`:
    1. Flattens the tensor into a rank-1 tensor.
    2. Pads the tensor to `d_2` dimensions with zeros, where `d_2` is the
       smallest power of 2 larger than or equal to `d`.
    3. Multiplies the padded tensor with random `+1/-1` values (i.e. flipping
       the signs). This is equivalent to multiplication by a diagonal matrix
       with Rademacher random varaibles on diagonal.
    4. Applies the fast Walsh-Hadamard transform.
  Steps 3 and 4 are repeated multiple times with independent randomness, if
  `num_repeats > 1`.

  The resulting tensors are passed to the `inner_agg_factory`. After
  aggregation, at `tff.SEREVR`, inverses of these steps are applied in reverse
  order.

  The allowed input dtypes are integers and floats. However, the dtype passed to
  the `inner_agg_factory` will always be a float.
  """

  def __init__(
      self,
      inner_agg_factory: Optional[factory.UnweightedAggregationFactory] = None,
      num_repeats: int = 1,
  ):
    if inner_agg_factory is None:
      inner_agg_factory = sum_factory.SumFactory()
    if not isinstance(inner_agg_factory, factory.UnweightedAggregationFactory):
      raise TypeError(
          'Provided `inner_agg_factory` must be an '
          f'UnweightedAggregationFactory. Found {type(inner_agg_factory)}.'
      )
    if not isinstance(num_repeats, int) or num_repeats < 1:
      raise ValueError(
          f'`num_repeats` should be a positive integer. Found {num_repeats}.'
      )
    self._inner_agg_factory = inner_agg_factory
    self._num_repeats = num_repeats

  def create(
      self, value_type: factory.ValueType
  ) -> aggregation_process.AggregationProcess:
    _check_value_type(value_type)
    value_specs = type_conversions.type_to_tf_tensor_specs(value_type)
    seeds_per_round = self._num_repeats * len(structure.flatten(value_type))
    next_global_seed_fn = _build_next_global_seed_fn(stride=seeds_per_round)

    @tensorflow_computation.tf_computation(value_type, SEED_TFF_TYPE)
    def client_transform(value, global_seed):
      @tf.function
      def transform(tensor, seed):
        for _ in range(self._num_repeats):
          tensor *= sample_rademacher(tf.shape(tensor), tensor.dtype, seed)
          tensor = tf.expand_dims(tensor, axis=0)
          tensor = hadamard.fast_walsh_hadamard_transform(tensor)
          tensor = tf.squeeze(tensor, axis=0)
          seed += 1
        return tensor

      value = _flatten_and_pad_zeros_pow2(value)
      seeds = _unique_seeds_for_struct(
          value, global_seed, stride=self._num_repeats
      )
      return tf.nest.map_structure(transform, value, seeds)

    inner_agg_process = self._inner_agg_factory.create(
        client_transform.type_signature.result
    )

    @tensorflow_computation.tf_computation(
        client_transform.type_signature.result, SEED_TFF_TYPE
    )
    def server_transform(value, global_seed):
      @tf.function
      def transform(tensor, seed):
        seed += self._num_repeats - 1
        for _ in range(self._num_repeats):
          tensor = tf.expand_dims(tensor, axis=0)
          tensor = hadamard.fast_walsh_hadamard_transform(tensor)
          tensor = tf.squeeze(tensor, axis=0)
          tensor *= sample_rademacher(tf.shape(tensor), tensor.dtype, seed)
          seed -= 1
        return tensor

      seeds = _unique_seeds_for_struct(
          value, global_seed, stride=self._num_repeats
      )
      value = tf.nest.map_structure(transform, value, seeds)
      return tf.nest.map_structure(
          _slice_and_reshape_to_template_spec, value, value_specs
      )

    @federated_computation.federated_computation()
    def init_fn():
      inner_state = inner_agg_process.initialize()
      my_state = intrinsics.federated_eval(
          tensorflow_computation.tf_computation(_init_global_seed),
          placements.SERVER,
      )
      return intrinsics.federated_zip((inner_state, my_state))

    @federated_computation.federated_computation(
        init_fn.type_signature.result, computation_types.at_clients(value_type)
    )
    def next_fn(state, value):
      next_fn_impl = _build_next_fn(
          client_transform,
          inner_agg_process,
          server_transform,
          next_global_seed_fn,
          'hd',
      )
      return next_fn_impl(state, value)

    return aggregation_process.AggregationProcess(init_fn, next_fn)


class DiscreteFourierTransformFactory(factory.UnweightedAggregationFactory):
  """`UnweightedAggregationFactory` for discrete Fourier transform.

  The created `tff.templates.AggregationProcess` takes an input structure
  and applies the randomized discrete Fourier transform (using TF's fast Fourier
  transform implementation `tf.signal.fft/ifft`) to each tensor in the
  structure, reshaped to a rank-1 tensor in `O(d*log(d))` time, where `d` is the
  number of elements of the tensor.

  See https://en.wikipedia.org/wiki/Discrete_Fourier_transform

  Specifically, for each tensor, the following operations are first performed at
  `tff.CLIENTS`:
    1. Flattens the tensor into a rank-1 tensor.
    2. Pads the tensor with zeros to an even number of elements (i.e. pad at
       most one zero).
    3. Packs the real valued tensor into a tensor with a complex dtype with
       `d/2` elements, by filling the real and imaginary values with two halves
       of the tensor.
    4. Randomly rotates each coordinate of the complex tensor.
    5. Applies the discrete Fourier transform.
    6. Unpacks the complex tensor back to a real tensor with length `d`.
    7. Normalizes the tensor by `1 / sqrt(d/2)`.
  Steps 4 and 5 are repeated multiple times with independent randomness, if
  `num_repeats > 1`.

  The resulting tensors are passed to the `inner_agg_factory`. After
  aggregation, at `tff.SEREVR`, inverses of these steps are applied in reverse
  order.

  The allowed input dtypes are integers and floats. However, the dtype passed to
  the `inner_agg_factory` will always be a float.
  """

  def __init__(
      self,
      inner_agg_factory: Optional[factory.UnweightedAggregationFactory] = None,
      num_repeats: int = 1,
  ):
    if inner_agg_factory is None:
      inner_agg_factory = sum_factory.SumFactory()
    if not isinstance(inner_agg_factory, factory.UnweightedAggregationFactory):
      raise TypeError(
          'Provided `inner_agg_factory` must be an'
          f'UnweightedAggregationFactory. Found {type(inner_agg_factory)}.'
      )
    if not isinstance(num_repeats, int) or num_repeats < 1:
      raise ValueError(
          f'`num_repeats` should be a positive integer. Found {num_repeats}.'
      )
    self._inner_agg_factory = inner_agg_factory
    self._num_repeats = num_repeats

  def create(
      self, value_type: factory.ValueType
  ) -> aggregation_process.AggregationProcess:
    _check_value_type(value_type)
    value_specs = type_conversions.structure_from_tensor_type_tree(
        lambda x: tf.TensorSpec(x.shape, x.dtype), value_type
    )
    seeds_per_round = self._num_repeats * len(structure.flatten(value_type))
    next_global_seed_fn = _build_next_global_seed_fn(stride=seeds_per_round)

    @tensorflow_computation.tf_computation(value_type, SEED_TFF_TYPE)
    def client_transform(value, global_seed):
      @tf.function
      def transform(tensor, seed):
        for _ in range(self._num_repeats):
          tensor = tf.reshape(tensor, [2, -1])
          tensor = tf.complex(real=tensor[0], imag=tensor[1])
          tensor *= sample_cis(tf.shape(tensor), seed, inverse=False)
          tensor = tf.signal.fft(tensor)
          tensor = tf.concat(
              [tf.math.real(tensor), tf.math.imag(tensor)], axis=0
          )
          tensor /= tf.cast(tf.sqrt(tf.size(tensor) / 2), OUTPUT_TF_DTYPE)
          seed += 1
        return tensor

      value = _flatten_and_pad_zeros_even(value)
      seeds = _unique_seeds_for_struct(
          value, global_seed, stride=self._num_repeats
      )
      return tf.nest.map_structure(transform, value, seeds)

    inner_agg_process = self._inner_agg_factory.create(
        client_transform.type_signature.result
    )

    @tensorflow_computation.tf_computation(
        client_transform.type_signature.result, SEED_TFF_TYPE
    )
    def server_transform(value, global_seed):
      @tf.function
      def transform(tensor, seed):
        seed += self._num_repeats - 1
        for _ in range(self._num_repeats):
          tensor *= tf.sqrt(tf.size(tensor, out_type=tensor.dtype) / 2.0)
          tensor = tf.reshape(tensor, [2, -1])
          tensor = tf.complex(real=tensor[0], imag=tensor[1])
          tensor = tf.signal.ifft(tensor)
          tensor *= sample_cis(tf.shape(tensor), seed, inverse=True)
          tensor = tf.concat(
              [tf.math.real(tensor), tf.math.imag(tensor)], axis=0
          )
          seed -= 1
        return tensor

      seeds = _unique_seeds_for_struct(
          value, global_seed, stride=self._num_repeats
      )
      value = tf.nest.map_structure(transform, value, seeds)
      return tf.nest.map_structure(
          _slice_and_reshape_to_template_spec, value, value_specs
      )

    @federated_computation.federated_computation()
    def init_fn():
      inner_state = inner_agg_process.initialize()
      my_state = intrinsics.federated_eval(
          tensorflow_computation.tf_computation(_init_global_seed),
          placements.SERVER,
      )
      return intrinsics.federated_zip((inner_state, my_state))

    @federated_computation.federated_computation(
        init_fn.type_signature.result, computation_types.at_clients(value_type)
    )
    def next_fn(state, value):
      next_fn_impl = _build_next_fn(
          client_transform,
          inner_agg_process,
          server_transform,
          next_global_seed_fn,
          'dft',
      )
      return next_fn_impl(state, value)

    return aggregation_process.AggregationProcess(init_fn, next_fn)


def _build_next_fn(
    client_transform, inner_agg_process, server_transform, update_my_state, name
):
  """Builds body of next_fn given components."""

  def next_fn_impl(state, value):
    inner_state, my_state = state
    client_my_state = intrinsics.federated_broadcast(my_state)
    projected_value = intrinsics.federated_map(
        client_transform, (value, client_my_state)
    )

    inner_agg_output = inner_agg_process.next(inner_state, projected_value)

    aggregate_value = intrinsics.federated_map(
        server_transform, (inner_agg_output.result, my_state)
    )

    new_state = (
        inner_agg_output.state,
        intrinsics.federated_map(update_my_state, my_state),
    )
    measurements = collections.OrderedDict(
        [(name, inner_agg_output.measurements)]
    )

    return measured_process.MeasuredProcessOutput(
        state=intrinsics.federated_zip(new_state),
        result=aggregate_value,
        measurements=intrinsics.federated_zip(measurements),
    )

  return next_fn_impl


def _init_global_seed():
  """Returns an initial global random seed.

  The seed is supposed to be used with `tf.random.stateless_*` ops. The first
  element corresponds to a timestamp drawn once, and the second element is a
  counter to be incremented (see `_build_next_global_seed_fn`) to ensure fresh
  randomness across invocations.

  Returns:
    A `tf.int64` tensor with shape [2].
  """
  microseconds_per_second = 10**6  # Timestamp returns fractional seconds.
  timestamp_microseconds = tf.cast(
      tf.timestamp() * microseconds_per_second, SEED_TF_DTYPE
  )
  return tf.convert_to_tensor([timestamp_microseconds, 0])


def _build_next_global_seed_fn(stride):
  """Builds function updating global random seed.

  The `stride` argument determines the increment to the counter in the random
  seed. This is used, when a single invocation of `next` function of an
  `IterativeProcess` uses the seed multiple times with increment of one, to make
  sure the same seed is not reused across rounds.

  Args:
    stride: An integer increment to the counter in the seed.

  Returns:
    A `tff.Computation` that takes and returns `tf.int64` tensor with shape [2].
  """

  @tensorflow_computation.tf_computation(SEED_TFF_TYPE)
  def _next_global_seed(global_seed):
    timestamp_microseconds, sequence_number = global_seed[0], global_seed[1]
    return tf.convert_to_tensor(
        [timestamp_microseconds, sequence_number + stride]
    )

  return _next_global_seed


def _unique_seeds_for_struct(struct, global_seed, stride):
  """Creates a structure of random seeds from a single random seed.

  This method primarily makes sure that the seeds are different, as the intent
  will be to use one `tf.random.stateless_*` invocation per element in the
  structure. The differences are done using the counter in the seed, incremented
  by `stride`. By default, `stride` can be `1`, but can be higher when each seed
  is to be reused multiple times with intermediate increments.

  Examples:
  ```
  _unique_seeds_for_struct(struct=(a, b, c), global_seed=(123, 1), stride=1)
  > ((123, 1), (123, 2), (123, 3))
  _unique_seeds_for_struct(struct=(a, b, c), global_seed=(123, 1), stride=4)
  > ((123, 1), (123, 5), (123, 9))
  ```

  Args:
    struct: A structure of elemnts compatible with `tf.nest`.
    global_seed: A `tf.int64` tensor with shape [2].
    stride: An integer increment between individual seeds in the returned
      structure of random seeds.

  Returns:
    A structure of random seeds (`tf.int64` tensors with shape [2]) matching
    the structure of `struct`.
  """
  num_tensors = len(tf.nest.flatten(struct))
  timestamp_microseconds, sequence_number = global_seed[0], global_seed[1]
  seeds = [
      tf.convert_to_tensor(
          [timestamp_microseconds, sequence_number + i * stride]
      )
      for i in range(num_tensors)
  ]
  return tf.nest.pack_sequence_as(struct, seeds)


def sample_rademacher(shape, dtype, seed):
  """Sample uniform random +1/-1 values with specified shape/dtype/seed."""
  rand_uniform = tf.random.stateless_uniform(shape=shape, seed=seed)
  return tf.cast(tf.sign(rand_uniform - 0.5), dtype)


def sample_cis(shape, seed, inverse=False):
  """Sample e^(i * theta) for theta in the range [0, 2pi] as tf.complex64."""
  # While it suffices to draw theta from [0, pi/2, pi, 3pi/2] (2 bits of
  # randomness) for each complex coordinate, sampling floating angles can avoid
  # the uniform integer sampler which may not be available on `tff.CLIENTS`.
  theta = tf.random.stateless_uniform(shape, seed, minval=0, maxval=2)
  theta *= math.pi
  theta *= tf.cond(tf.cast(inverse, tf.bool), lambda: -1.0, lambda: 1.0)
  return tf.exp(tf.complex(real=0.0, imag=theta))


def _slice_and_reshape_to_template_spec(x, original_spec):
  """Unpad, reshape, and cast a component tensor after backward transform."""
  original_len = original_spec.shape.num_elements()
  reshaped_x = tf.reshape(x[:original_len], original_spec.shape)
  if original_spec.dtype.is_integer:
    reshaped_x = tf.round(reshaped_x)
  return tf.cast(reshaped_x, original_spec.dtype)


def _flatten_and_pad_zeros_pow2(struct):
  struct = tf.nest.map_structure(
      lambda x: tf.reshape(tf.cast(x, OUTPUT_TF_DTYPE), [-1]), struct
  )
  return tf.nest.map_structure(_pad_zeros_pow2, struct)


def _flatten_and_pad_zeros_even(struct):
  struct = tf.nest.map_structure(
      lambda x: tf.reshape(tf.cast(x, OUTPUT_TF_DTYPE), [-1]), struct
  )
  return tf.nest.map_structure(_pad_zeros_even, struct)


def _pad_zeros_pow2(x):
  """Pads a rank-1 tensor with zeros to the next power of two dimensions."""
  size = tf.size(x)
  log2_size = tf.math.log(tf.cast(size, tf.float32)) / math.log(2.0)
  # NOTE: We perform `pow` in float32 to avoid the integer TF `pow` op,
  # improving runtimes that use selective op registration.
  pad_size = tf.cast(2.0 ** tf.math.ceil(log2_size), tf.int32)
  return tf.concat([x, tf.zeros([pad_size - size], x.dtype)], axis=0)


def _pad_zeros_even(x):
  """Pads a rank-1 tensor with zeros to the next even dimensions."""
  num_zeros = tf.cast(tf.equal(tf.size(x) % 2, 1), tf.int32)
  return tf.concat([x, tf.zeros([num_zeros], x.dtype)], axis=0)


def _check_value_type(value_type):
  """Check value_type meets documented criteria."""
  if not (
      value_type.is_tensor()
      or (
          value_type.is_struct_with_python()
          and type_analysis.is_structure_of_tensors(value_type)
      )
  ):
    raise TypeError(
        'Expected `value_type` to be `TensorType` or '
        '`StructWithPythonType` containing only `TensorType`. '
        f'Found type: {repr(value_type)}'
    )

  if not (
      type_analysis.is_structure_of_floats(value_type)
      or type_analysis.is_structure_of_integers(value_type)
  ):
    raise TypeError(
        'Component dtypes of `value_type` must be all integers or '
        f'all floats. Found {value_type}.'
    )
