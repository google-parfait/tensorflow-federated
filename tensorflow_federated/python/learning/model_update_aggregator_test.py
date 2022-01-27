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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.mapreduce import form_utils
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning import model_update_aggregator

_float_type = computation_types.TensorType(tf.float32)
_float_matrix_type = computation_types.TensorType(tf.float32, [200, 300])


class ModelUpdateAggregatorTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple', False, False, False),
      ('zeroing', True, False, False),
      ('clipping', False, True, False),
      ('zeroing_and_clipping', True, True, False),
      ('debug_measurements', False, False, True),
      ('zeroing_clipping_debug_measurements', True, True, True),
  )
  def test_robust_aggregator_weighted(self, zeroing, clipping,
                                      add_debug_measurements):
    factory_ = model_update_aggregator.robust_aggregator(
        zeroing=zeroing,
        clipping=clipping,
        add_debug_measurements=add_debug_measurements)

    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(_float_type, _float_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertTrue(process.is_weighted)

  @parameterized.named_parameters(
      ('simple', False, False, False),
      ('zeroing', True, False, False),
      ('clipping', False, True, False),
      ('zeroing_and_clipping', True, True, False),
      ('debug_measurements', False, False, True),
      ('zeroing_clipping_debug_measurements', True, True, True),
  )
  def test_robust_aggregator_unweighted(self, zeroing, clipping,
                                        add_debug_measurements):
    factory_ = model_update_aggregator.robust_aggregator(
        zeroing=zeroing,
        clipping=clipping,
        weighted=False,
        add_debug_measurements=add_debug_measurements)

    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create(_float_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertFalse(process.is_weighted)

  @parameterized.named_parameters(
      ('simple', False),
      ('zeroing', True),
  )
  def test_dp_aggregator(self, zeroing):
    factory_ = model_update_aggregator.dp_aggregator(
        noise_multiplier=1e-2, clients_per_round=10, zeroing=zeroing)

    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create(_float_type)
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
        zeroing=zeroing, clipping=clipping)

    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(_float_type, _float_type)
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
        zeroing=zeroing, clipping=clipping, weighted=False)

    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create(_float_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertFalse(process.is_weighted)

  def test_weighted_secure_aggregator_only_contains_secure_aggregation(self):
    aggregator = model_update_aggregator.secure_aggregator(
        weighted=True).create(_float_matrix_type, _float_type)
    try:
      static_assert.assert_not_contains_unsecure_aggregation(aggregator.next)
    except:  # pylint: disable=bare-except
      self.fail('Secure aggregator contains non-secure aggregation.')

  def test_unweighted_secure_aggregator_only_contains_secure_aggregation(self):
    aggregator = model_update_aggregator.secure_aggregator(
        weighted=False).create(_float_matrix_type)
    try:
      static_assert.assert_not_contains_unsecure_aggregation(aggregator.next)
    except:  # pylint: disable=bare-except
      self.fail('Secure aggregator contains non-secure aggregation.')

  @parameterized.named_parameters(
      ('simple', False, False, False),
      ('zeroing', True, False, False),
      ('clipping', False, True, False),
      ('zeroing_and_clipping', True, True, False),
      ('debug_measurements', False, False, True),
      ('zeroing_clipping_debug_measurements', True, True, True),
  )
  def test_compression_aggregator_weighted(self, zeroing, clipping,
                                           add_debug_measurements):
    factory_ = model_update_aggregator.compression_aggregator(
        zeroing=zeroing,
        clipping=clipping,
        add_debug_measurements=add_debug_measurements)

    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(_float_type, _float_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertTrue(process.is_weighted)

  @parameterized.named_parameters(
      ('simple', False, False, False),
      ('zeroing', True, False, False),
      ('clipping', False, True, False),
      ('zeroing_and_clipping', True, True, False),
      ('debug_measurements', False, False, True),
      ('zeroing_clipping_debug_measurements', True, True, True),
  )
  def test_compression_aggregator_unweighted(self, zeroing, clipping,
                                             add_debug_measurements):
    factory_ = model_update_aggregator.compression_aggregator(
        zeroing=zeroing,
        clipping=clipping,
        weighted=False,
        add_debug_measurements=add_debug_measurements)

    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create(_float_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertFalse(process.is_weighted)


class CompilerIntegrationTest(test_case.TestCase, parameterized.TestCase):
  """Integration tests making sure compiler does not end up confused.

  These tests compile the aggregator into MapReduceForm and check that the
  amount of aggregated scalars roughly match the expectations, thus making sure
  the compiler does not accidentally introduce duplicates.
  """

  def _check_aggregated_scalar_count(self,
                                     aggregator,
                                     max_scalars,
                                     min_scalars=0):
    aggregator = _mrfify_aggregator(aggregator)
    mrf = form_utils.get_map_reduce_form_for_iterative_process(aggregator)
    num_aggregated_scalars = type_analysis.count_tensors_in_type(
        mrf.work.type_signature.result)['parameters']
    self.assertLess(num_aggregated_scalars, max_scalars)
    self.assertGreaterEqual(num_aggregated_scalars, min_scalars)
    return mrf

  def test_robust_aggregator(self):
    aggregator = model_update_aggregator.robust_aggregator().create(
        _float_matrix_type, _float_type)
    self._check_aggregated_scalar_count(aggregator, 60000 * 1.01, 60000)

  def test_dp_aggregator(self):
    aggregator = model_update_aggregator.dp_aggregator(
        0.01, 10).create(_float_matrix_type)
    self._check_aggregated_scalar_count(aggregator, 60000 * 1.01, 60000)

  def test_secure_aggregator(self):
    aggregator = model_update_aggregator.secure_aggregator().create(
        _float_matrix_type, _float_type)
    mrf = self._check_aggregated_scalar_count(aggregator, 60000 * 1.01, 60000)

    # The MapReduceForm should be using secure aggregation.
    self.assertTrue(mrf.securely_aggregates_tensors)

  def test_compression_aggregator(self):
    aggregator = model_update_aggregator.compression_aggregator().create(
        _float_matrix_type, _float_type)
    # Default compression should reduce the size aggregated by more than 60%.
    self._check_aggregated_scalar_count(aggregator, 60000 * 0.4)


def _mrfify_aggregator(aggregator):
  """Makes aggregator compatible with MapReduceForm."""

  if aggregator.is_weighted:

    @computations.federated_computation(
        aggregator.next.type_signature.parameter[0],
        computation_types.at_clients(
            (aggregator.next.type_signature.parameter[1].member,
             aggregator.next.type_signature.parameter[2].member)))
    def next_fn(state, value):
      output = aggregator.next(state, value[0], value[1])
      return output.state, intrinsics.federated_zip(
          (output.result, output.measurements))
  else:

    @computations.federated_computation(aggregator.next.type_signature.parameter
                                       )
    def next_fn(state, value):
      output = aggregator.next(state, value)
      return output.state, intrinsics.federated_zip(
          (output.result, output.measurements))

  return iterative_process.IterativeProcess(aggregator.initialize, next_fn)


if __name__ == '__main__':
  test_case.main()
