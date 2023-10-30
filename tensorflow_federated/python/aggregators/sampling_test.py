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

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import sampling
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_test_utils

# Convenience type aliases.
FunctionType = computation_types.FunctionType
SequenceType = computation_types.SequenceType
StructType = computation_types.StructType
StructWithPythonType = computation_types.StructWithPythonType
TensorType = computation_types.TensorType

# Type for the random seed used in sampling is int64 tensor with shape [2].
SEED_TYPE = computation_types.TensorType(tf.int64, shape=[2])
TEST_SEED = 42
RANDOM_VALUE_TYPE = computation_types.TensorType(tf.int32, [None])


class BuildReservoirTypeTest(tf.test.TestCase):

  def test_scalar(self):
    self.assertEqual(
        sampling._build_reservoir_type(TensorType(tf.float32)),
        StructWithPythonType(
            [
                ('random_seed', SEED_TYPE),
                ('random_values', RANDOM_VALUE_TYPE),
                ('samples', TensorType(tf.float32, [None])),
            ],
            collections.OrderedDict,
        ),
    )

  def test_structure_of_tensors(self):
    self.assertEqual(
        sampling._build_reservoir_type(
            computation_types.to_type(
                collections.OrderedDict(
                    a=TensorType(tf.float32),
                    b=[TensorType(tf.int64, [2]), TensorType(tf.bool)],
                )
            )
        ),
        StructWithPythonType(
            [
                ('random_seed', SEED_TYPE),
                ('random_values', RANDOM_VALUE_TYPE),
                (
                    'samples',
                    collections.OrderedDict(
                        a=TensorType(tf.float32, [None]),
                        b=[
                            TensorType(tf.int64, [None, 2]),
                            TensorType(tf.bool, [None]),
                        ],
                    ),
                ),
            ],
            collections.OrderedDict,
        ),
    )

  def test_fails_non_tensor_or_struct_with_python_type(self):
    with self.assertRaises(TypeError):
      sampling._build_reservoir_type(SequenceType(TensorType(tf.float32, [3])))
    with self.assertRaises(TypeError):
      sampling._build_reservoir_type(
          StructType(elements=[(None, TensorType(tf.float32, [3]))])
      )


class BuildInitialSampleReservoirTest(tf.test.TestCase):

  def assertAllEqual(self, *args, **kwargs):
    # `test_case.assertAllEqual` doesn't handle nested structures in a
    # convenient way (np.ndarray != tf.Tensor with the same values). We change
    # this to use assertAllClose with zero value tolerances.
    self.assertAllClose(*args, **kwargs, atol=0.0, rtol=0.0)

  def test_scalar(self):
    with self.subTest('fixed_seed'):
      initial_reservoir = sampling._build_initial_sample_reservoir(
          TensorType(tf.int32), seed=TEST_SEED
      )
      self.assertAllEqual(
          initial_reservoir,
          collections.OrderedDict(
              random_seed=tf.convert_to_tensor((TEST_SEED, TEST_SEED)),
              random_values=tf.zeros([0], tf.int32),
              samples=tf.zeros([0], dtype=tf.int32),
          ),
      )
    with self.subTest('no_seed'):
      initial_reservoir = sampling._build_initial_sample_reservoir(
          TensorType(tf.int32)
      )
      self.assertLen(initial_reservoir['random_seed'], 2)
      self.assertEqual(
          initial_reservoir['random_seed'][0],
          initial_reservoir['random_seed'][1],
      )
      self.assertEqual(
          initial_reservoir['random_seed'][0], sampling.SEED_SENTINEL
      )

  def test_structure_of_tensors(self):
    value_type = computation_types.to_type(
        collections.OrderedDict(
            a=TensorType(tf.float32),
            b=[TensorType(tf.int64, [2]), TensorType(tf.bool)],
        )
    )
    initial_reservoir = sampling._build_initial_sample_reservoir(
        sample_value_type=value_type, seed=TEST_SEED
    )
    self.assertAllEqual(
        initial_reservoir,
        collections.OrderedDict(
            random_seed=tf.convert_to_tensor((TEST_SEED, TEST_SEED)),
            random_values=tf.zeros([0], tf.int32),
            samples=collections.OrderedDict(
                a=tf.zeros([0], dtype=tf.float32),
                b=[
                    tf.zeros([0, 2], dtype=tf.int64),
                    tf.zeros([0], dtype=tf.bool),
                ],
            ),
        ),
    )

  def test_fails_with_non_tensor_type(self):
    with self.assertRaises(TypeError):
      sampling._build_initial_sample_reservoir(
          sample_value_type=SequenceType(TensorType(tf.int32)), seed=TEST_SEED
      )
    with self.assertRaises(TypeError):
      sampling._build_initial_sample_reservoir(
          sample_value_type=computation_types.to_type(
              collections.OrderedDict(a=SequenceType(TensorType(tf.int32)))
          ),
          seed=TEST_SEED,
      )


class BuildSampleValueComputationTest(tf.test.TestCase):

  def assertAllEqual(self, *args, **kwargs):
    # `test_case.assertAllEqual` doesn't handle nested structures in a
    # convenient way (np.ndarray != tf.Tensor with the same values). We change
    # this to use assertAllClose with zero value tolerances.
    self.assertAllClose(*args, **kwargs, atol=0.0, rtol=0.0)

  def test_scalar_random_seed(self):
    example_type = TensorType(tf.int32)
    sample_computation = sampling._build_sample_value_computation(
        example_type, sample_size=1
    )
    reservoir_type = sampling._build_reservoir_type(example_type)
    expected_type = FunctionType(
        parameter=collections.OrderedDict(
            reservoir=reservoir_type, sample=example_type
        ),
        result=reservoir_type,
    )
    type_test_utils.assert_types_identical(
        sample_computation.type_signature, expected_type
    )
    # Get the sentinel seed so that the first call initializes based on
    # timestamp.
    reservoir = sampling._build_initial_sample_reservoir(example_type)
    self.assertAllEqual(
        reservoir['random_seed'],
        [sampling.SEED_SENTINEL, sampling.SEED_SENTINEL],
    )
    reservoir = sample_computation(reservoir, 1)
    # The first value of the seed was the timestamp, it should be greater than
    # 1_600_000_000_000 (September 2020) and within 60 seconds of now.
    self.assertGreater(reservoir['random_seed'][0], 1_600_000_000_000)
    self.assertLess(
        tf.cast(tf.timestamp() * 1000.0, tf.int64)
        - reservoir['random_seed'][0],
        60,
    )
    # The second value should we a random number. We assert its not the
    # sentinel, though it ccould be with probability 1 / 2**32.
    self.assertNotEqual(reservoir['random_seed'][1], sampling.SEED_SENTINEL)

  def test_scalar_fixed_seed(self):
    example_type = TensorType(tf.int32)
    sample_computation = sampling._build_sample_value_computation(
        example_type, sample_size=1
    )
    reservoir_type = sampling._build_reservoir_type(example_type)
    expected_type = FunctionType(
        parameter=collections.OrderedDict(
            reservoir=reservoir_type, sample=example_type
        ),
        result=reservoir_type,
    )
    type_test_utils.assert_types_identical(
        sample_computation.type_signature, expected_type
    )
    reservoir = sampling._build_initial_sample_reservoir(
        example_type, seed=TEST_SEED
    )
    reservoir = sample_computation(reservoir, 1)
    self.assertAllEqual(
        reservoir,
        collections.OrderedDict(
            random_seed=(TEST_SEED, 100565241),
            random_values=[100565241],
            samples=[1],
        ),
    )
    # New value was not sampled, its random value was too low, but it
    # changes the seed for the next iteration.
    reservoir = sample_computation(reservoir, 2)
    self.assertAllEqual(
        reservoir,
        collections.OrderedDict(
            random_seed=(TEST_SEED, -1479562987),
            random_values=[100565241],
            samples=[1],
        ),
    )
    # The PRNG doesn't generate a number for sampling until 5th example.
    for i in range(3, 6):
      reservoir = sample_computation(reservoir, i)
    self.assertAllEqual(
        reservoir,
        collections.OrderedDict(
            random_seed=(TEST_SEED, 756274747),
            random_values=[756274747],
            samples=[5],
        ),
    )

  def test_structure_of_tensors(self):
    example_type = computation_types.to_type(
        collections.OrderedDict(
            a=TensorType(tf.int32, [3]),
            b=[TensorType(tf.float32), TensorType(tf.bool)],
        )
    )
    sample_computation = sampling._build_sample_value_computation(
        example_type, sample_size=1
    )
    reservoir_type = sampling._build_reservoir_type(example_type)
    expected_type = FunctionType(
        parameter=collections.OrderedDict(
            reservoir=reservoir_type, sample=example_type
        ),
        result=reservoir_type,
    )
    type_test_utils.assert_types_identical(
        sample_computation.type_signature, expected_type
    )
    reservoir = sampling._build_initial_sample_reservoir(
        example_type, seed=TEST_SEED
    )
    reservoir = sample_computation(
        reservoir, collections.OrderedDict(a=[0, 1, 2], b=[1.0, True])
    )
    expected_sample = collections.OrderedDict(a=[[0, 1, 2]], b=[[1.0], [True]])
    self.assertAllEqual(
        reservoir,
        collections.OrderedDict(
            random_seed=(TEST_SEED, 100565241),
            random_values=[100565241],
            samples=expected_sample,
        ),
    )
    # New value was not sampled, its random value was too low, but it
    # changes the seed for the next iteration.
    reservoir = sample_computation(
        reservoir, collections.OrderedDict(a=[3, 4, 5], b=[2.0, False])
    )
    self.assertAllEqual(
        reservoir,
        collections.OrderedDict(
            random_seed=(TEST_SEED, -1479562987),
            random_values=[100565241],
            samples=expected_sample,
        ),
    )
    # The PRNG doesn't generate a number for sampling until 5th example.
    for i in range(3, 6):
      reservoir = sample_computation(
          reservoir,
          collections.OrderedDict(a=list(range(i, i + 3)), b=[float(i), False]),
      )
    self.assertAllEqual(
        reservoir,
        collections.OrderedDict(
            random_seed=(TEST_SEED, 756274747),
            random_values=[756274747],
            samples=collections.OrderedDict(a=[[5, 6, 7]], b=[[5.0], [False]]),
        ),
    )


class BuildMergeSamplesComputationTest(tf.test.TestCase):

  def assertAllEqual(self, *args, **kwargs):
    # `test_case.assertAllEqual` doesn't handle nested structures in a
    # convenient way (different numerical arrays are not equal, e.g. np.ndarray
    # != tf.Tensor with the same values). We change this to use assertAllClose
    # with zero value tolerances.
    self.assertAllClose(*args, **kwargs, atol=0.0, rtol=0.0)

  def test_scalar(self):
    example_type = TensorType(tf.int32)
    merge_computation = sampling._build_merge_samples_computation(
        example_type, sample_size=5
    )
    reservoir_type = sampling._build_reservoir_type(example_type)
    expected_type = FunctionType(
        parameter=collections.OrderedDict(a=reservoir_type, b=reservoir_type),
        result=reservoir_type,
    )
    type_test_utils.assert_types_identical(
        merge_computation.type_signature, expected_type
    )
    reservoir_a = sampling._build_initial_sample_reservoir(
        example_type, seed=TEST_SEED
    )
    reservoir_a['random_values'] = [1, 3, 5]
    reservoir_a['samples'] = [3, 9, 15]
    with self.subTest('downsample'):
      reservoir_b = sampling._build_initial_sample_reservoir(
          example_type, seed=TEST_SEED + 1
      )
      reservoir_b['random_values'] = [2, 4, 6, 8]
      reservoir_b['samples'] = [6, 12, 18, 24]
      merged_reservoir = merge_computation(reservoir_a, reservoir_b)
      self.assertAllEqual(
          merged_reservoir,
          collections.OrderedDict(
              # Arbitrarily take seeds from `a`, discarded later.
              random_seed=tf.convert_to_tensor((TEST_SEED, TEST_SEED)),
              random_values=[3, 5, 4, 6, 8],
              samples=[9, 15, 12, 18, 24],
          ),
      )
    with self.subTest('keep_all'):
      reservoir_b = sampling._build_initial_sample_reservoir(
          example_type, seed=TEST_SEED + 1
      )
      reservoir_b['random_values'] = [2]
      reservoir_b['samples'] = [6]
      # We select the value from reservoir_b because its random_value was
      # higher.
      merged_reservoir = merge_computation(reservoir_a, reservoir_b)
      self.assertAllEqual(
          merged_reservoir,
          collections.OrderedDict(
              # Arbitrarily take seeds from `a`, discarded later.
              random_seed=tf.convert_to_tensor((TEST_SEED, TEST_SEED)),
              random_values=[1, 3, 5, 2],
              samples=[3, 9, 15, 6],
          ),
      )
    with self.subTest('tie_breakers'):
      # In case of tie, we take the as many values from `a` first.
      reservoir_b = sampling._build_initial_sample_reservoir(
          example_type, seed=TEST_SEED
      )
      reservoir_b['random_values'] = [5, 5, 5, 5, 5]  # all tied with `a`
      reservoir_b['samples'] = [-1, -1, -1, -1, -1]
      merged_reservoir = merge_computation(reservoir_a, reservoir_b)
      self.assertAllEqual(
          merged_reservoir,
          collections.OrderedDict(
              random_seed=tf.convert_to_tensor((TEST_SEED, TEST_SEED)),
              random_values=[5, 5, 5, 5, 5],
              samples=[15, -1, -1, -1, -1],
          ),
      )

  def test_structure_of_tensors(self):
    example_type = computation_types.to_type(
        collections.OrderedDict(
            a=TensorType(tf.int32, [3]),
            b=[TensorType(tf.float32), TensorType(tf.bool)],
        )
    )
    merge_computation = sampling._build_merge_samples_computation(
        example_type, sample_size=5
    )
    reservoir_type = sampling._build_reservoir_type(example_type)
    expected_type = FunctionType(
        parameter=collections.OrderedDict(a=reservoir_type, b=reservoir_type),
        result=reservoir_type,
    )
    type_test_utils.assert_types_identical(
        merge_computation.type_signature, expected_type
    )
    reservoir_a = sampling._build_initial_sample_reservoir(
        example_type, seed=TEST_SEED
    )
    reservoir_a['random_values'] = [1, 3, 5]
    reservoir_a['samples'] = collections.OrderedDict(
        a=[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        b=[[0.0, 1.0, 2.0], [True, False, True]],
    )
    with self.subTest('downsample'):
      reservoir_b = sampling._build_initial_sample_reservoir(
          example_type, seed=TEST_SEED + 1
      )
      reservoir_b['random_values'] = [2, 4, 6, 8]
      reservoir_b['samples'] = collections.OrderedDict(
          a=[[0, -1, -2], [-1, -2, -3], [-2, -3, -4], [-3, -4, -5]],
          b=[[-1.0, -2.0, -3.0, -4.0], [True, False, False, True]],
      )
      merged_reservoir = merge_computation(reservoir_a, reservoir_b)
      self.assertAllEqual(
          merged_reservoir,
          collections.OrderedDict(
              # Arbitrarily take seeds from `a`, discarded later.
              random_seed=tf.convert_to_tensor((TEST_SEED, TEST_SEED)),
              random_values=[3, 5, 4, 6, 8],
              samples=collections.OrderedDict(
                  a=[
                      [1, 2, 3],
                      [2, 3, 4],
                      [-1, -2, -3],
                      [-2, -3, -4],
                      [-3, -4, -5],
                  ],
                  b=[
                      [1.0, 2.0, -2.0, -3.0, -4.0],
                      [False, True, False, False, True],
                  ],
              ),
          ),
      )
    with self.subTest('keep_all'):
      reservoir_b = sampling._build_initial_sample_reservoir(
          example_type, seed=TEST_SEED + 1
      )
      reservoir_b['random_values'] = [2]
      reservoir_b['samples'] = collections.OrderedDict(
          a=[[0, -1, -2]], b=[[-1.0], [True]]
      )
      # We select the value from reservoir_b because its random_value was
      # higher.
      merged_reservoir = merge_computation(reservoir_a, reservoir_b)
      self.assertAllEqual(
          merged_reservoir,
          collections.OrderedDict(
              # Arbitrarily take seeds from `a`, discarded later.
              random_seed=tf.convert_to_tensor((TEST_SEED, TEST_SEED)),
              random_values=[1, 3, 5, 2],
              samples=collections.OrderedDict(
                  a=[[0, 1, 2], [1, 2, 3], [2, 3, 4], [-0, -1, -2]],
                  b=[[0.0, 1.0, 2.0, -1.0], [True, False, True, True]],
              ),
          ),
      )
    with self.subTest('tie_breakers'):
      # In case of tie, we take the as many values from `a` first.
      reservoir_b = sampling._build_initial_sample_reservoir(
          example_type, seed=TEST_SEED
      )
      reservoir_b['random_values'] = [5, 5, 5, 5, 5]  # all tied with `a`
      reservoir_b['samples'] = collections.OrderedDict(
          a=[[-1, -1, -1]] * 5, b=[[-1] * 5, [False] * 5]
      )
      merged_reservoir = merge_computation(reservoir_a, reservoir_b)
      self.assertAllEqual(
          merged_reservoir,
          collections.OrderedDict(
              random_seed=tf.convert_to_tensor((TEST_SEED, TEST_SEED)),
              random_values=[5, 5, 5, 5, 5],
              samples=collections.OrderedDict(
                  a=[[2, 3, 4]] + [[-1, -1, -1]] * 4,
                  b=[[2] + [-1] * 4, [True] + [False] * 4],
              ),
          ),
      )


class BuildFinalizeSampleTest(tf.test.TestCase):

  def assertAllEqual(self, *args, **kwargs):
    # `test_case.assertAllEqual` doesn't handle nested structures in a
    # convenient way (different numerical arrays are not equal, e.g. np.ndarray
    # != tf.Tensor with the same values). We change this to use assertAllClose
    # with zero value tolerances.
    self.assertAllClose(*args, **kwargs, atol=0.0, rtol=0.0)

  def test_scalar(self):
    example_type = computation_types.to_type(TensorType(tf.int32))
    finalize_computation = sampling._build_finalize_sample_computation(
        example_type
    )
    reservoir_type = sampling._build_reservoir_type(example_type)
    expected_type = FunctionType(
        parameter=reservoir_type, result=reservoir_type.samples
    )
    type_test_utils.assert_types_identical(
        finalize_computation.type_signature, expected_type
    )
    reservoir = sampling._build_initial_sample_reservoir(
        example_type, seed=TEST_SEED
    )
    reservoir['random_values'] = [3, 5, 7]
    test_samples = [3, 9, 27]
    reservoir['samples'] = test_samples
    self.assertAllEqual(finalize_computation(reservoir), test_samples)

  def test_structure(self):
    example_type = computation_types.to_type(
        collections.OrderedDict(
            a=TensorType(tf.int32),
            b=[TensorType(tf.float32, [3]), TensorType(tf.bool)],
        )
    )
    finalize_computation = sampling._build_finalize_sample_computation(
        example_type
    )
    reservoir_type = sampling._build_reservoir_type(example_type)
    expected_type = FunctionType(
        parameter=reservoir_type, result=reservoir_type.samples
    )
    type_test_utils.assert_types_identical(
        finalize_computation.type_signature, expected_type
    )
    reservoir = sampling._build_initial_sample_reservoir(
        example_type, seed=TEST_SEED
    )
    reservoir['random_values'] = [3, 5, 7]
    test_samples = collections.OrderedDict(
        a=[3, 9, 27], b=[[[0, 1, 2], [1, 2, 3], [2, 3, 4]], [True, False, True]]
    )
    reservoir['samples'] = test_samples
    self.assertAllEqual(finalize_computation(reservoir), test_samples)


class BuildCheckNonFiniteLeavesComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('float32_nan', tf.float32, np.nan, True),
      ('bfloat16_inf', tf.bfloat16, np.inf, True),
      ('half_nan', tf.half, np.nan, True),
      ('float64_inf', tf.float64, np.inf, True),
      ('int32_finite', tf.int32, 1, False),
      ('bool_finite', tf.bool, False, False),
  )
  def test_scalar(self, dtype, value, is_non_finite):
    computation = sampling._build_check_non_finite_leaves_computation(
        TensorType(dtype)
    )
    result = computation(value)
    expected_result = tf.constant(is_non_finite, dtype=tf.int64)
    self.assertEqual(result, expected_result)

  def test_structure(self):
    value_type = computation_types.to_type(
        collections.OrderedDict(
            a=TensorType(tf.int32),
            b=[TensorType(tf.float32, [3]), TensorType(tf.bool)],
            c=collections.OrderedDict(d=TensorType(tf.float64, [2, 2])),
        )
    )
    computation = sampling._build_check_non_finite_leaves_computation(
        value_type
    )
    value = collections.OrderedDict(
        a=1,
        b=[[1.0, np.nan, np.inf], True],
        c=collections.OrderedDict(d=[[np.inf, 2.0], [3.0, 4.0]]),
    )
    result = computation(value)
    expected_result = collections.OrderedDict(
        a=tf.constant(0, dtype=tf.int64),
        b=[tf.constant(1, dtype=tf.int64), tf.constant(0, dtype=tf.int64)],
        c=collections.OrderedDict(d=tf.constant(1, dtype=tf.int64)),
    )
    self.assertEqual(result, expected_result)

  def test_fails_with_non_tensor_type(self):
    with self.assertRaisesRegex(TypeError, 'only contain `TensorType`s'):
      sampling._build_check_non_finite_leaves_computation(
          SequenceType(TensorType(tf.int32))
      )
    with self.assertRaisesRegex(TypeError, 'only contain `TensorType`s'):
      sampling._build_check_non_finite_leaves_computation(
          computation_types.to_type(
              collections.OrderedDict(
                  a=TensorType(tf.float32, [3]),
                  b=[SequenceType(TensorType(tf.int32)), TensorType(tf.bool)],
              )
          )
      )


class UnweightedReservoirSamplingFactoryTest(
    tf.test.TestCase, parameterized.TestCase
):

  def test_create(self):
    factory = sampling.UnweightedReservoirSamplingFactory(sample_size=10)
    with self.subTest('scalar_aggregator'):
      factory.create(computation_types.to_type(tf.int32))
    with self.subTest('structure_aggregator'):
      factory.create(
          computation_types.to_type(
              collections.OrderedDict(
                  a=TensorType(tf.int32),
                  b=[TensorType(tf.float32, [3]), TensorType(tf.bool)],
              )
          )
      )

  @parameterized.named_parameters(('two_samples', 2), ('four_samples', 4))
  def test_sample_size_limits(self, sample_size):
    process = sampling.UnweightedReservoirSamplingFactory(
        sample_size=sample_size
    ).create(computation_types.to_type(tf.int32))
    state = process.initialize()
    output = process.next(
        state,
        # Create a 2  * sample_size values from clients.
        tf.random.stateless_uniform(
            shape=(sample_size * 2,),
            minval=None,
            seed=tf.convert_to_tensor((TEST_SEED, TEST_SEED)),
            dtype=tf.int32,
        )
        .numpy()
        .tolist(),
    )
    self.assertEqual(output.result.shape, (sample_size,))

  def test_unfilled_reservoir(self):
    process = sampling.UnweightedReservoirSamplingFactory(sample_size=4).create(
        computation_types.to_type(tf.int32)
    )
    state = process.initialize()
    # Create 3 client values to aggregate.
    client_values = (
        tf.random.stateless_uniform(
            shape=(3,),
            minval=None,
            seed=tf.convert_to_tensor((TEST_SEED, TEST_SEED)),
            dtype=tf.int32,
        )
        .numpy()
        .tolist()
    )
    output = process.next(state, client_values)
    self.assertCountEqual(output.result, client_values)

  def test_build_factory_fails_invalid_argument(self):
    with self.assertRaises(ValueError):
      sampling.UnweightedReservoirSamplingFactory(sample_size=0)
    with self.assertRaises(ValueError):
      sampling.UnweightedReservoirSamplingFactory(sample_size=-1)
    with self.assertRaises(TypeError):
      sampling.UnweightedReservoirSamplingFactory(sample_size=None)
    with self.assertRaises(TypeError):
      sampling.UnweightedReservoirSamplingFactory(sample_size='5')

  def test_measurements_scalar_value(self):
    process = sampling.UnweightedReservoirSamplingFactory(sample_size=1).create(
        computation_types.to_type(tf.float32)
    )
    state = process.initialize()
    output = process.next(state, [1.0, np.nan, np.inf, 2.0, 3.0])
    # Two clients' values are non-infinte.
    self.assertEqual(output.measurements, tf.constant(2, dtype=tf.int64))

  def test_measurements_structure_value(self):
    process = sampling.UnweightedReservoirSamplingFactory(sample_size=1).create(
        computation_types.to_type(
            collections.OrderedDict(
                a=TensorType(tf.float32),
                b=[TensorType(tf.float32, [2, 2]), TensorType(tf.bool)],
            )
        )
    )
    state = process.initialize()
    output = process.next(
        state,
        [
            collections.OrderedDict(
                a=1.0, b=[[[1.0, np.nan], [np.inf, 4.0]], True]
            ),
            collections.OrderedDict(a=2.0, b=[[[1.0, 2.0], [3.0, 4.0]], False]),
            collections.OrderedDict(
                a=np.inf, b=[[[np.nan, 2.0], [3.0, 4.0]], True]
            ),
        ],
    )
    self.assertEqual(
        output.measurements,
        collections.OrderedDict(
            # One client has non-infinte tensors for this leaf node.
            a=tf.constant(1, dtype=tf.int64),
            # Two clients have non-infinte tensors for this leaf node.
            b=[tf.constant(2, dtype=tf.int64), tf.constant(0, dtype=tf.int64)],
        ),
    )


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
