# Copyright 2020, The TensorFlow Federated Authors.
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


from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.backends.mapreduce import form_utils
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning import debug_measurements
from tensorflow_federated.python.learning import model_update_aggregator

_FLOAT_TYPE = computation_types.TensorType(np.float32)
_FLOAT_MATRIX_TYPE = computation_types.TensorType(np.float32, [200, 300])


class ModelUpdateAggregatorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple', False, False, None),
      ('zeroing', True, False, None),
      ('clipping', False, True, None),
      ('zeroing_and_clipping', True, True, None),
      (
          'debug_measurements',
          False,
          False,
          debug_measurements.add_debug_measurements,
      ),
      (
          'zeroing_clipping_debug_measurements',
          True,
          True,
          debug_measurements.add_debug_measurements,
      ),
  )
  def test_robust_aggregator_weighted(
      self, zeroing, clipping, debug_measurements_fn
  ):
    factory_ = model_update_aggregator.robust_aggregator(
        zeroing=zeroing,
        clipping=clipping,
        debug_measurements_fn=debug_measurements_fn,
    )

    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(_FLOAT_TYPE, _FLOAT_TYPE)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertTrue(process.is_weighted)

  @parameterized.named_parameters(
      ('simple', False, False, None),
      ('zeroing', True, False, None),
      ('clipping', False, True, None),
      ('zeroing_and_clipping', True, True, None),
      (
          'debug_measurements',
          False,
          False,
          debug_measurements.add_debug_measurements,
      ),
      (
          'zeroing_clipping_debug_measurements',
          True,
          True,
          debug_measurements.add_debug_measurements,
      ),
  )
  def test_robust_aggregator_unweighted(
      self, zeroing, clipping, debug_measurements_fn
  ):
    factory_ = model_update_aggregator.robust_aggregator(
        zeroing=zeroing,
        clipping=clipping,
        weighted=False,
        debug_measurements_fn=debug_measurements_fn,
    )

    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create(_FLOAT_TYPE)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertFalse(process.is_weighted)

  @parameterized.named_parameters(
      (
          'debug_measurements',
          False,
          False,
          debug_measurements.add_debug_measurements_with_mixed_dtype,
      ),
      (
          'zeroing_clipping_debug_measurements',
          True,
          True,
          debug_measurements.add_debug_measurements_with_mixed_dtype,
      ),
  )
  def test_robust_aggregator_weighted_mixed_dtype(
      self, zeroing, clipping, debug_measurements_fn
  ):
    factory_ = model_update_aggregator.robust_aggregator(
        zeroing=zeroing,
        clipping=clipping,
        debug_measurements_fn=debug_measurements_fn,
    )

    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(_FLOAT_TYPE, _FLOAT_TYPE)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertTrue(process.is_weighted)

  def test_wrong_debug_measurements_fn_robust_aggregator(self):
    """Expect error if debug_measurements_fn is wrong."""
    with self.assertRaises(TypeError):

      def wrong_debug_measurements_fn(
          aggregation_factory: factory.AggregationFactory,
      ) -> ...:
        del aggregation_factory
        return (
            debug_measurements._calculate_client_update_statistics_mixed_dtype(
                [1.0], [1.0]
            )
        )

      model_update_aggregator.robust_aggregator(
          debug_measurements_fn=wrong_debug_measurements_fn
      )

  @parameterized.named_parameters(
      ('simple', False),
      ('zeroing', True),
  )
  def test_dp_aggregator(self, zeroing):
    factory_ = model_update_aggregator.dp_aggregator(
        noise_multiplier=1e-2, clients_per_round=10, zeroing=zeroing
    )

    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create(_FLOAT_TYPE)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertFalse(process.is_weighted)

  @parameterized.named_parameters(
      ('simple', False, False),
      ('zeroing', True, False),
      ('clipping', False, True),
      ('zeroing_and_clipping', True, True),
  )
  def test_secure_aggregator_weighted(self, zeroing, clipping):
    factory_ = model_update_aggregator.secure_aggregator(
        zeroing=zeroing, clipping=clipping
    )

    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(_FLOAT_TYPE, _FLOAT_TYPE)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertTrue(process.is_weighted)

  @parameterized.named_parameters(
      ('simple', False, False),
      ('zeroing', True, False),
      ('clipping', False, True),
      ('zeroing_and_clipping', True, True),
  )
  def test_secure_aggregator_unweighted(self, zeroing, clipping):
    factory_ = model_update_aggregator.secure_aggregator(
        zeroing=zeroing, clipping=clipping, weighted=False
    )

    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create(_FLOAT_TYPE)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertFalse(process.is_weighted)

  def test_weighted_secure_aggregator_only_contains_secure_aggregation(self):
    aggregator = model_update_aggregator.secure_aggregator(
        weighted=True
    ).create(_FLOAT_MATRIX_TYPE, _FLOAT_TYPE)
    static_assert.assert_not_contains_unsecure_aggregation(aggregator.next)

  def test_unweighted_secure_aggregator_only_contains_secure_aggregation(self):
    aggregator = model_update_aggregator.secure_aggregator(
        weighted=False
    ).create(_FLOAT_MATRIX_TYPE)
    static_assert.assert_not_contains_unsecure_aggregation(aggregator.next)

  def test_ddp_secure_aggregator_only_contains_secure_aggregation(self):
    aggregator = model_update_aggregator.ddp_secure_aggregator(
        noise_multiplier=1e-2, expected_clients_per_round=10
    ).create(_FLOAT_MATRIX_TYPE)
    static_assert.assert_not_contains_unsecure_aggregation(aggregator.next)

  @parameterized.named_parameters(
      ('zeroing_float', True, _FLOAT_TYPE),
      ('zeroing_float_matrix', True, _FLOAT_MATRIX_TYPE),
      ('no_zeroing_float', False, _FLOAT_TYPE),
      ('no_zeroing_float_matrix', False, _FLOAT_MATRIX_TYPE),
  )
  def test_ddp_secure_aggregator_unweighted(self, zeroing, dtype):
    aggregator = model_update_aggregator.ddp_secure_aggregator(
        noise_multiplier=1e-2,
        expected_clients_per_round=10,
        bits=16,
        zeroing=zeroing,
    )

    self.assertIsInstance(aggregator, factory.UnweightedAggregationFactory)
    process = aggregator.create(dtype)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertFalse(process.is_weighted)

  @parameterized.named_parameters(
      ('simple', False, False, None),
      ('zeroing', True, False, None),
      ('clipping', False, True, None),
      ('zeroing_and_clipping', True, True, None),
      (
          'debug_measurements',
          False,
          False,
          debug_measurements.add_debug_measurements,
      ),
      (
          'zeroing_clipping_debug_measurements',
          True,
          True,
          debug_measurements.add_debug_measurements,
      ),
  )
  def test_compression_aggregator_weighted(
      self, zeroing, clipping, debug_measurements_fn
  ):
    factory_ = model_update_aggregator.compression_aggregator(
        zeroing=zeroing,
        clipping=clipping,
        debug_measurements_fn=debug_measurements_fn,
    )

    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(_FLOAT_TYPE, _FLOAT_TYPE)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertTrue(process.is_weighted)

  @parameterized.named_parameters(
      ('simple', False, False, None),
      ('zeroing', True, False, None),
      ('clipping', False, True, None),
      ('zeroing_and_clipping', True, True, None),
      (
          'debug_measurements',
          False,
          False,
          debug_measurements.add_debug_measurements,
      ),
      (
          'zeroing_clipping_debug_measurements',
          True,
          True,
          debug_measurements.add_debug_measurements,
      ),
  )
  def test_compression_aggregator_unweighted(
      self, zeroing, clipping, debug_measurements_fn
  ):
    factory_ = model_update_aggregator.compression_aggregator(
        zeroing=zeroing,
        clipping=clipping,
        weighted=False,
        debug_measurements_fn=debug_measurements_fn,
    )

    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create(_FLOAT_TYPE)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertFalse(process.is_weighted)

  @parameterized.named_parameters(
      (
          'debug_measurements',
          False,
          False,
          debug_measurements.add_debug_measurements_with_mixed_dtype,
      ),
      (
          'zeroing_clipping_debug_measurements',
          True,
          True,
          debug_measurements.add_debug_measurements_with_mixed_dtype,
      ),
  )
  def test_compression_aggregator_weighted_mixed_dtype(
      self, zeroing, clipping, debug_measurements_fn
  ):
    factory_ = model_update_aggregator.compression_aggregator(
        zeroing=zeroing,
        clipping=clipping,
        debug_measurements_fn=debug_measurements_fn,
    )

    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(_FLOAT_TYPE, _FLOAT_TYPE)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertTrue(process.is_weighted)

  def test_wrong_debug_measurements_fn_compression_aggregator(self):
    """Expect error if debug_measurements_fn is wrong."""
    with self.assertRaises(TypeError):

      def wrong_debug_measurements_fn(
          aggregation_factory: factory.AggregationFactory,
      ) -> ...:
        del aggregation_factory
        return (
            debug_measurements._calculate_client_update_statistics_mixed_dtype(
                [1.0], [1.0]
            )
        )

      model_update_aggregator.compression_aggregator(
          debug_measurements_fn=wrong_debug_measurements_fn
      )


class CompilerIntegrationTest(parameterized.TestCase):
  """Integration tests making sure compiler does not end up confused.

  These tests compile the aggregator into MapReduceForm and check that the
  amount of aggregated scalars roughly match the expectations, thus making sure
  the compiler does not accidentally introduce duplicates.
  """

  def _check_aggregated_scalar_count(
      self, aggregator, max_scalars, min_scalars=0
  ):
    aggregator = _mrfify_aggregator(aggregator)
    mrf = form_utils.get_map_reduce_form_for_computation(aggregator.next)
    num_aggregated_scalars = type_analysis.count_tensors_in_type(
        mrf.work.type_signature.result
    )['parameters']
    self.assertLess(num_aggregated_scalars, max_scalars)
    self.assertGreaterEqual(num_aggregated_scalars, min_scalars)
    return mrf

  def test_robust_aggregator(self):
    aggregator = model_update_aggregator.robust_aggregator().create(
        _FLOAT_MATRIX_TYPE, _FLOAT_TYPE
    )
    self._check_aggregated_scalar_count(aggregator, 60000 * 1.01, 60000)

  def test_dp_aggregator(self):
    aggregator = model_update_aggregator.dp_aggregator(0.01, 10).create(
        _FLOAT_MATRIX_TYPE
    )
    self._check_aggregated_scalar_count(aggregator, 60000 * 1.01, 60000)

  def test_secure_aggregator(self):
    aggregator = model_update_aggregator.secure_aggregator().create(
        _FLOAT_MATRIX_TYPE, _FLOAT_TYPE
    )
    mrf = self._check_aggregated_scalar_count(aggregator, 60000 * 1.01, 60000)

    # The MapReduceForm should be using secure aggregation.
    self.assertTrue(mrf.securely_aggregates_tensors)

  def test_compression_aggregator(self):
    aggregator = model_update_aggregator.compression_aggregator().create(
        _FLOAT_MATRIX_TYPE, _FLOAT_TYPE
    )
    # Default compression should reduce the size aggregated by more than 60%.
    self._check_aggregated_scalar_count(aggregator, 60000 * 0.4)

  def test_ddp_secure_aggregator(self):
    self.skipTest('b/305747127')
    aggregator = model_update_aggregator.ddp_secure_aggregator(
        noise_multiplier=1e-2, expected_clients_per_round=10
    ).create(_FLOAT_MATRIX_TYPE)
    # The Hadmard transform requires padding to next power of 2
    mrf = self._check_aggregated_scalar_count(aggregator, 2**16 * 1.01, 60000)

    # The MapReduceForm should be using secure aggregation.
    self.assertTrue(mrf.securely_aggregates_tensors)


def _mrfify_aggregator(aggregator):
  """Makes aggregator compatible with MapReduceForm."""

  if aggregator.is_weighted:

    @federated_computation.federated_computation(
        aggregator.next.type_signature.parameter[0],
        computation_types.FederatedType(
            (
                aggregator.next.type_signature.parameter[1].member,
                aggregator.next.type_signature.parameter[2].member,
            ),
            placements.CLIENTS,
        ),
    )
    def next_fn(state, value):
      output = aggregator.next(state, value[0], value[1])
      return output.state, intrinsics.federated_zip(
          (output.result, output.measurements)
      )

  else:

    @federated_computation.federated_computation(
        aggregator.next.type_signature.parameter
    )
    def next_fn(state, value):
      output = aggregator.next(state, value)
      return output.state, intrinsics.federated_zip(
          (output.result, output.measurements)
      )

  return iterative_process.IterativeProcess(aggregator.initialize, next_fn)


if __name__ == '__main__':
  absltest.main()
