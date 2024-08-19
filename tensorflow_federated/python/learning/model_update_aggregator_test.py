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
      (
          'debug_measurements',
          False,
          False,
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
