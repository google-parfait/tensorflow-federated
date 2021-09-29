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

import abc
import collections
import functools
import math
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te

SEED_TF_TYPE = tf.int64
OUTPUT_TF_TYPE = tf.float32


# TODO(b/192618450): Revisit the abc design.
class _FlatteningFactory(factory.UnweightedAggregationFactory, abc.ABC):
  """`UnweightedAggregationFactory` for flattening values across dimensions.

  The created `tff.templates.AggregationProcess` takes an input tensor structure
  and applies a random basis transform to each tensor reshaped as a vector.
  This, for instance, can be implemented as a random unitary transform on the
  vector that can be geometrically interpreted as randomly rotating/reflecting
  the vector.

  The useful property is that the transform can spread out the coordinate
  values more uniformly across the vector dimensions. This is useful for
  many down-stream operations, such as:
    1. Uniform quantization, where the resulting vector rotation/reflection
       reduces the dynamic range of the coordinates to be quantized, decreasing
       the error incurred by quantization; and
    2. Modular wrapping, where the spread-out coordinates after the transform
       could lead to less modular wrapping.

  Specifically, for any component tensor of the input tensor structrure, the
  forward transform would correspond to the following operations (specific
  implementation depends on the transform):
    1. Reshapes the tensor into a vector (rank-1 tensor).
    2. Pads the vector with zeros (the exact number of zeros to pad depend on
       the specific transform).
    3. Applies a random basis transform to the vector (the randomness depends
       on the transform).
  The backward transform reverts the above steps.

  The forward transform is applied on `tff.CLIENTS` while the inverse transform
  (reverting the forward transform) is applied on `tff.SERVER`.

  This factory only accepts `value_type` of either `tff.TensorType` or
  `tff.StructWithPythonType` and expects the dtype of component tensors to be
  either all real integers or all real floats, and it will otherwise raise an
  error.
  """

  ######## Abstract methods: transform-dependent ########
  @abc.abstractmethod
  def _preprocess_tensor(self, x):
    """Preprocess a component tensor for forward transform."""
    raise NotImplementedError

  @abc.abstractmethod
  def _forward_transform_vector(self, x, seed_pair, num_repeats):
    raise NotImplementedError

  @abc.abstractmethod
  def _backward_transform_vector(self, x, seed_pair, num_repeats):
    raise NotImplementedError

  ######## Concrete methods: shared across different transforms ########
  def __init__(
      self,
      inner_agg_factory: Optional[factory.UnweightedAggregationFactory] = None,
      num_repeats: int = 1):
    """Initializes the FlatteningFactory.

    Args:
      inner_agg_factory: The inner `UnweightedAggregationFactory` to aggregate
        the values after the transform.
      num_repeats: The number of times to repeat the transform on each component
        tensor. Must be a positive integer.

    Raises:
      TypeError: If `inner_agg_factory` is not an instance of
        `tff.aggregators.UnweightedAggregationFactory`
      ValueError: If `num_repeats` is not a positive integer.
    """
    if inner_agg_factory is None:
      inner_agg_factory = sum_factory.SumFactory()

    if not isinstance(inner_agg_factory, factory.UnweightedAggregationFactory):
      raise TypeError('`inner_agg_factory` must have type '
                      'UnweightedAggregationFactory. '
                      f'Found {type(inner_agg_factory)}.')

    if not isinstance(num_repeats, int) or num_repeats < 1:
      raise ValueError('`num_repeats` should be a positive integer. '
                       f'Found {num_repeats}.')

    self._inner_agg_factory = inner_agg_factory
    self._num_repeats = num_repeats

  def _postprocess_tensor(self, x, original_spec):
    """Unpad, reshape, and cast a component tensor after backward transform."""
    original_len = original_spec.shape.num_elements()
    reshaped_x = tf.reshape(x[:original_len], original_spec.shape)
    if original_spec.dtype.is_integer:
      reshaped_x = tf.round(reshaped_x)
    return tf.cast(reshaped_x, original_spec.dtype)

  def _forward_transform_struct(self, struct, seed_pair, num_repeats):
    """Applies the transform to each component tensor of a structure."""
    num_tensors = len(tf.nest.flatten(struct))
    seed_pairs_list = [seed_pair + i * num_repeats for i in range(num_tensors)]
    seed_pairs_struct = tf.nest.pack_sequence_as(struct, seed_pairs_list)

    transform_fn = functools.partial(
        self._forward_transform_vector, num_repeats=num_repeats)
    prep_struct = tf.nest.map_structure(self._preprocess_tensor, struct)
    return tf.nest.map_structure(transform_fn, prep_struct, seed_pairs_struct)

  def _backward_transform_struct(self, struct, struct_py_type, seed_pair,
                                 num_repeats):
    """Inverts the transform applied to the structure."""
    num_tensors = len(tf.nest.flatten(struct))
    seed_pairs_list = [seed_pair + i * num_repeats for i in range(num_tensors)]
    seed_pairs_struct = tf.nest.pack_sequence_as(struct, seed_pairs_list)

    transform_fn = functools.partial(
        self._backward_transform_vector, num_repeats=num_repeats)
    unrotated_struct = tf.nest.map_structure(transform_fn, struct,
                                             seed_pairs_struct)
    return tf.nest.map_structure(self._postprocess_tensor, unrotated_struct,
                                 struct_py_type)

  def create(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:
    # As the Hadamard transform alters the tensor specs, we compute the Python
    # structure of the types for the inverse transform.
    if (value_type.is_struct_with_python() and
        type_analysis.is_structure_of_tensors(value_type)):
      py_type = type_conversions.structure_from_tensor_type_tree(
          lambda x: tf.TensorSpec(x.shape, x.dtype), value_type)
    elif value_type.is_tensor():
      py_type = tf.TensorSpec(value_type.shape, value_type.dtype)
    else:
      raise TypeError('Expected `value_type` to be `TensorType` or '
                      '`StructWithPythonType` containing only `TensorType`. '
                      f'Found type: {repr(value_type)}')

    _check_component_dtypes(value_type)
    seed_pair_type = init_seed_pair.type_signature.result

    @computations.tf_computation(value_type, seed_pair_type)
    def forward_transform_structure(value, seed_pair):
      return self._forward_transform_struct(value, seed_pair, self._num_repeats)

    @computations.tf_computation(
        forward_transform_structure.type_signature.result, seed_pair_type)
    def backward_transform_structure(value, seed_pair):
      return self._backward_transform_struct(value, py_type, seed_pair,
                                             self._num_repeats)

    tff_inner_type = forward_transform_structure.type_signature.result
    inner_agg_process = self._inner_agg_factory.create(tff_inner_type)

    @computations.federated_computation()
    def init_fn():
      state = collections.OrderedDict(
          round_seed=intrinsics.federated_eval(init_seed_pair,
                                               placements.SERVER),
          inner_agg_process=inner_agg_process.initialize())
      return intrinsics.federated_zip(state)

    @computations.federated_computation(init_fn.type_signature.result,
                                        computation_types.at_clients(value_type)
                                       )
    def next_fn(state, value):
      server_seed = state['round_seed']
      client_seed = intrinsics.federated_broadcast(server_seed)
      rotated_value = intrinsics.federated_map(forward_transform_structure,
                                               (value, client_seed))

      inner_state = state['inner_agg_process']
      inner_agg_output = inner_agg_process.next(inner_state, rotated_value)

      unrotated_agg_value = intrinsics.federated_map(
          backward_transform_structure, (inner_agg_output.result, server_seed))

      new_state = collections.OrderedDict(
          round_seed=intrinsics.federated_map(next_seed_pair, server_seed),
          inner_agg_process=inner_agg_output.state)
      measurements = collections.OrderedDict(
          rotation=inner_agg_output.measurements)

      return measured_process.MeasuredProcessOutput(
          state=intrinsics.federated_zip(new_state),
          result=unrotated_agg_value,
          measurements=intrinsics.federated_zip(measurements))

    return aggregation_process.AggregationProcess(init_fn, next_fn)


class HadamardTransformFactory(_FlatteningFactory):
  """Implements `FlatteningFactory` with the Fast Walsh-Hadamard Transform.

  The created `tff.templates.AggregationProcess` takes an input tensor structure
  and applies the randomized fast Walsh-Hadamard transform to each tensor
  flattened as a vector in O(d*log(d)) time, where `d` is the vector dimension.
  https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform

  Specifically, for any component tensor, the forward transform corresponds to
  the following operations at `tff.CLIENTS`:
    1. Flattens the tensor into a vector (rank-1 tensor).
    2. Pads the vector to `d_2` dimensions with zeros, where `d_2` is
       the smallest power of 2 larger than or equal to the number of elements.
    3. Multiplies the padded vector with random +1/-1 values (i.e. flipping the
       signs of the vector). This corresponds to applying a random orthogonal
       diagonal matrix.
    4. Multiplies the randomly flipped vector the `d_2 x d_2` Hadamard matrix.
  The backward transform reverts the above steps at `tff.SERVER`.

  The side effects of this aggregator include:
    1. The component tensors are flattened and padded with zeros. For example,
       if the factory receives a `value_type` of <float32[3],float32[2,3]>,
       the inner factory will operate on <float32[4],float32[8]>.
    2. The dtype of the component tensors are casted to floats.
  """

  def _preprocess_tensor(self, x):
    # Casts, reshapes, and pads to a float vector with power-of-2 dimensions.
    return pad_zeros_pow2(tf.reshape(tf.cast(x, OUTPUT_TF_TYPE), [-1]))

  @tf.function
  def _forward_transform_vector(self, x, seed_pair, num_repeats):

    def _transform_fn(x, cur_seed_pair):
      # Randomly flip signs.
      signs = sample_rademacher(tf.shape(x), x.dtype, cur_seed_pair)
      flipped_x = signs * x
      # Apply Hadamard matrix.
      expanded_x = tf.expand_dims(flipped_x, axis=0)
      rotated_x = te.utils.fast_walsh_hadamard_transform(expanded_x)
      return tf.squeeze(rotated_x, axis=0)

    for index in range(num_repeats):
      x = _transform_fn(x, seed_pair + index)
    return x

  @tf.function
  def _backward_transform_vector(self, x, seed_pair, num_repeats):

    def _transform_fn(x, cur_seed_pair):
      expanded_x = tf.expand_dims(x, axis=0)
      unrotated_x = te.utils.fast_walsh_hadamard_transform(expanded_x)
      unrotated_x = tf.squeeze(unrotated_x, axis=0)
      signs = sample_rademacher(tf.shape(unrotated_x), x.dtype, cur_seed_pair)
      return signs * unrotated_x

    for index in range(num_repeats - 1, -1, -1):
      x = _transform_fn(x, seed_pair + index)
    return x


class DiscreteFourierTransformFactory(_FlatteningFactory):
  """Implements `FlatteningFactory` with the Discrete Fourier Transform.

  The created `tff.templates.AggregationProcess` takes an input tensor structure
  and applies the randomized discrete Fourier transform (using TF's fast Fourier
  transform implementation `tf.signal.fft/ifft`) to each tensor flattened as
  a vector in O(d*log(d)) time, where `d` is the vector dimension.
  https://en.wikipedia.org/wiki/Discrete_Fourier_transform

  Specifically, for any component tensor, the forward transform corresponds to
  the following operations at `tff.CLIENTS`:
    1. Flattens the tensor into a vector (rank-1 tensor).
    2. Pads the vector to an even number of dimensions `d` with zeros (i.e. pad
       at most one zero).
    3. Packs the real vector into a complex vector with length `d/2` by filling
       the real and imaginary values with two halves of the real vector.
    4. Multiplies the each coordinate of the complex vector with random
       rotations (i.e. apply `cos(x) + isin(x)` for x in the range [0, 2pi]).
    5. Multiplies the resulting vector with the `d x d` DFT matrix.
    6. Unpacks the complex vector back to a real vector with length `d`.
    7. Normalize the vector by `1 / sqrt(d/2)`.
  The backward transform reverts the above steps at `tff.SERVER`.

  The side effects of this aggregator include:
    1. The component tensors are flattened and padded with zeros. For example,
       if the factory receives a `value_type` of <float32[3],float32[2,3]>,
       the inner factory will operate on <float32[4],float32[6]>.
    2. The dtype of the component tensors are casted to floats.
  """

  def _preprocess_tensor(self, x):
    """Casts, reshapes, and pads to a float vector with even dimensions."""
    return pad_zeros_even(tf.reshape(tf.cast(x, OUTPUT_TF_TYPE), [-1]))

  @tf.function
  def _forward_transform_vector(self, x, seed_pair, num_repeats):

    def _transform_fn(x, cur_seed_pair):
      split_x = tf.reshape(x, [2, -1])
      complex_x = tf.complex(real=split_x[0], imag=split_x[1])
      # Apply randomness.
      complex_x *= sample_cis(tf.shape(complex_x), cur_seed_pair)
      # Apply FFT as rotation.
      fft_x = tf.signal.fft(complex_x)
      rotated_x = tf.concat([tf.math.real(fft_x), tf.math.imag(fft_x)], axis=0)
      # Normalize by 1/sqrt(d/2) where `d` is the padded dim to an even number.
      return rotated_x / tf.cast(tf.sqrt(tf.size(rotated_x) / 2), tf.float32)

    for index in range(num_repeats):
      x = _transform_fn(x, seed_pair + index)
    return x

  @tf.function
  def _backward_transform_vector(self, x, seed_pair, num_repeats):

    def _transform_fn(x, cur_seed_pair):
      unnorm_x = x * tf.cast(tf.sqrt(tf.size(x) / 2), tf.float32)
      split_x = tf.reshape(unnorm_x, [2, -1])
      complex_x = tf.complex(real=split_x[0], imag=split_x[1])
      # Invert the FFT rotation.
      ifft_x = tf.signal.ifft(complex_x)
      # Undo randomness.
      ifft_x *= sample_cis(tf.shape(ifft_x), cur_seed_pair, inverse=True)
      return tf.concat([tf.math.real(ifft_x), tf.math.imag(ifft_x)], axis=0)

    for index in range(num_repeats - 1, -1, -1):
      x = _transform_fn(x, seed_pair + index)
    return x


@computations.tf_computation()
def init_seed_pair():
  microseconds_per_second = 10**6  # Timestamp returns fractional seconds.
  timestamp_microseconds = tf.cast(tf.timestamp() * microseconds_per_second,
                                   SEED_TF_TYPE)
  return tf.convert_to_tensor([timestamp_microseconds, 0])


@computations.tf_computation(init_seed_pair.type_signature.result)
def next_seed_pair(seed_pair):
  timestamp_microseconds, sequence_number = seed_pair[0], seed_pair[1]
  return tf.convert_to_tensor([timestamp_microseconds, sequence_number + 1])


def sample_rademacher(shape, dtype, seed_pair):
  """Sample uniform random +1/-1 values with specified shape/dtype/seed_pair."""
  rand_uniform = tf.random.stateless_uniform(shape=shape, seed=seed_pair)
  return tf.cast(tf.sign(rand_uniform - 0.5), dtype)


def sample_cis(shape, seed_pair, inverse=False):
  """Sample e^(i * theta) for theta in the range [0, 2pi] as tf.complex64."""
  # While it suffices to draw theta from [0, pi/2, pi, 3pi/2] (2 bits of
  # randomness) for each complex coordinate, sampling floating angles can avoid
  # the uniform integer sampler which may not be available on `tff.CLIENTS`.
  theta = tf.random.stateless_uniform(shape, seed_pair, minval=0, maxval=2)
  theta *= math.pi
  theta *= tf.cond(tf.cast(inverse, tf.bool), lambda: -1.0, lambda: 1.0)
  return tf.exp(tf.complex(real=0.0, imag=theta))


def pad_zeros_pow2(x):
  """Pads a rank-1 tensor with zeros to the next power of two dimensions."""
  size = tf.size(x)
  log2_size = tf.math.log(tf.cast(size, tf.float32)) / math.log(2.0)
  # NOTE: We perform `pow` in float32 to avoid the integer TF `pow` op which is
  # currently not available in the pruning graph. This can be avoided via
  # Grappler's constant folding optimizer, but it must be disabled due to
  # b/164455653. While float32 can only represent the nonnegative integer range
  # [0, 2^24] exactly, we only consider powers of 2 for padding and thus can
  # tolerate up to 2^30 with a cast to int32.
  pad_size = tf.cast(2.0**tf.math.ceil(log2_size), tf.int32)
  return pad_zeros(x, pad_size - size)


def pad_zeros_even(x):
  """Pads a rank-1 tensor with zeros to the next even dimensions."""
  num_zeros = tf.cast(tf.equal(tf.size(x) % 2, 1), tf.int32)
  return pad_zeros(x, num_zeros)


def pad_zeros(x, num_zeros):
  """Pads a rank-1 tensor with shape (d,) with `num_zero` zeros."""
  tf.debugging.assert_rank(x, 1, f'Expected rank-1 tensors, but found {x}.')
  return tf.pad(x, [[0, tf.maximum(0, num_zeros)]])


def _check_component_dtypes(value_type):
  """Checks all components of the `value_type` to be either ints or floats."""
  if not (type_analysis.is_structure_of_floats(value_type) or
          type_analysis.is_structure_of_integers(value_type)):
    raise TypeError('Component dtypes of `value_type` must all be integers or '
                    f'floats. Found {value_type}.')
