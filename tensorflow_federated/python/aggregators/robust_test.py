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

import collections
import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import aggregator_test_utils
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import robust
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process

_test_struct_type = [(np.float32, (3,)), np.float32]


def _make_test_struct_value(x):
  return [tf.constant(x, dtype=tf.float32, shape=(3,)), x]


def _clipped_mean(clip=2.0):
  return robust.clipping_factory(clip, mean.MeanFactory())


def _clipped_sum(clip=2.0):
  return robust.clipping_factory(clip, sum_factory.SumFactory())


def _zeroed_mean(clip=2.0, norm_order=2.0):
  return robust.zeroing_factory(clip, mean.MeanFactory(), norm_order)


def _zeroed_sum(clip=2.0, norm_order=2.0):
  return robust.zeroing_factory(clip, sum_factory.SumFactory(), norm_order)


_float_at_server = computation_types.FederatedType(
    computation_types.TensorType(np.float32), placements.SERVER
)
_float_at_clients = computation_types.FederatedType(
    computation_types.TensorType(np.float32), placements.CLIENTS
)


@federated_computation.federated_computation()
def _test_init_fn():
  return intrinsics.federated_value(1.0, placements.SERVER)


@federated_computation.federated_computation(
    _float_at_server, _float_at_clients
)
def _test_next_fn(state, value):
  del value
  return intrinsics.federated_map(
      tensorflow_computation.tf_computation(lambda x: x + 1.0, np.float32),
      state,
  )


@federated_computation.federated_computation(_float_at_server)
def _test_report_fn(state):
  return state


def _test_norm_process(
    init_fn=_test_init_fn, next_fn=_test_next_fn, report_fn=_test_report_fn
):
  return estimation_process.EstimationProcess(init_fn, next_fn, report_fn)


class ClippingFactoryComputationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('float', np.float32),
      ('struct', _test_struct_type),
  )
  def test_clip_type_properties_simple(self, value_type):
    factory = _clipped_sum()
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    server_state_type = computation_types.FederatedType(
        collections.OrderedDict(
            clipping_norm=(), inner_agg=(), clipped_count_agg=()
        ),
        placements.SERVER,
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(
            clipping=(),
            clipping_norm=robust.NORM_TF_TYPE,
            clipped_count=robust.COUNT_TF_TYPE,
        ),
        placements.SERVER,
    )
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.FederatedType(
                value_type, placements.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.FederatedType(
                value_type, placements.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  @parameterized.named_parameters(
      ('float_value_float32_weight', np.float32, np.float32),
      ('struct_value_float32_weight', _test_struct_type, np.float32),
      ('float_value_float64_weight', np.float32, np.float64),
      ('struct_value_float64_weight', _test_struct_type, np.float64),
  )
  def test_clip_type_properties_weighted(self, value_type, weight_type):
    factory = _clipped_mean()
    value_type = computation_types.to_type(value_type)
    weight_type = computation_types.to_type(weight_type)
    process = factory.create(value_type, weight_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    mean_state_type = collections.OrderedDict(
        value_sum_process=(), weight_sum_process=()
    )
    server_state_type = computation_types.FederatedType(
        collections.OrderedDict(
            clipping_norm=(), inner_agg=mean_state_type, clipped_count_agg=()
        ),
        placements.SERVER,
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(
            clipping=collections.OrderedDict(mean_value=(), mean_weight=()),
            clipping_norm=robust.NORM_TF_TYPE,
            clipped_count=robust.COUNT_TF_TYPE,
        ),
        placements.SERVER,
    )
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.FederatedType(
                value_type, placements.CLIENTS
            ),
            weight=computation_types.FederatedType(
                weight_type, placements.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.FederatedType(
                value_type, placements.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  @parameterized.named_parameters(
      ('float', np.float32),
      ('struct', _test_struct_type),
  )
  def test_zero_type_properties_simple(self, value_type):
    factory = _zeroed_sum()
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    server_state_type = computation_types.FederatedType(
        collections.OrderedDict(
            zeroing_norm=(), inner_agg=(), zeroed_count_agg=()
        ),
        placements.SERVER,
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(
            zeroing=(),
            zeroing_norm=robust.NORM_TF_TYPE,
            zeroed_count=robust.COUNT_TF_TYPE,
        ),
        placements.SERVER,
    )
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.FederatedType(
                value_type, placements.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.FederatedType(
                value_type, placements.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  @parameterized.named_parameters(
      ('float_value_float32_weight', np.float32, np.float32),
      ('struct_value_float32_weight', _test_struct_type, np.float32),
      ('float_value_float64_weight', np.float32, np.float64),
      ('struct_value_float64_weight', _test_struct_type, np.float64),
  )
  def test_zero_type_properties_weighted(self, value_type, weight_type):
    factory = _zeroed_mean()
    value_type = computation_types.to_type(value_type)
    weight_type = computation_types.to_type(weight_type)
    process = factory.create(value_type, weight_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    mean_state_type = collections.OrderedDict(
        value_sum_process=(), weight_sum_process=()
    )
    server_state_type = computation_types.FederatedType(
        collections.OrderedDict(
            zeroing_norm=(), inner_agg=mean_state_type, zeroed_count_agg=()
        ),
        placements.SERVER,
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(
            zeroing=collections.OrderedDict(mean_value=(), mean_weight=()),
            zeroing_norm=robust.NORM_TF_TYPE,
            zeroed_count=robust.COUNT_TF_TYPE,
        ),
        placements.SERVER,
    )
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.FederatedType(
                value_type, placements.CLIENTS
            ),
            weight=computation_types.FederatedType(
                weight_type, placements.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.FederatedType(
                value_type, placements.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  @parameterized.named_parameters(
      ('float', np.float32),
      ('struct', _test_struct_type),
  )
  def test_zero_type_properties_with_zeroed_count_agg_factory(self, value_type):
    factory = robust.zeroing_factory(
        zeroing_norm=1.0,
        inner_agg_factory=sum_factory.SumFactory(),
        norm_order=2.0,
        zeroed_count_sum_factory=aggregator_test_utils.SumPlusOneFactory(),
    )
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    server_state_type = computation_types.FederatedType(
        collections.OrderedDict(
            zeroing_norm=(), inner_agg=(), zeroed_count_agg=np.int32
        ),
        placements.SERVER,
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(
            zeroing=(),
            zeroing_norm=robust.NORM_TF_TYPE,
            zeroed_count=robust.COUNT_TF_TYPE,
        ),
        placements.SERVER,
    )
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.FederatedType(
                value_type, placements.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.FederatedType(
                value_type, placements.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  @parameterized.named_parameters(
      ('float', np.float32),
      ('struct', _test_struct_type),
  )
  def test_clip_type_properties_with_clipped_count_agg_factory(
      self, value_type
  ):
    factory = robust.clipping_factory(
        clipping_norm=1.0,
        inner_agg_factory=sum_factory.SumFactory(),
        clipped_count_sum_factory=aggregator_test_utils.SumPlusOneFactory(),
    )
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    server_state_type = computation_types.FederatedType(
        collections.OrderedDict(
            clipping_norm=(), inner_agg=(), clipped_count_agg=np.int32
        ),
        placements.SERVER,
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(
            clipping=(),
            clipping_norm=robust.NORM_TF_TYPE,
            clipped_count=robust.COUNT_TF_TYPE,
        ),
        placements.SERVER,
    )
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.FederatedType(
                value_type, placements.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.FederatedType(
                value_type, placements.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  @parameterized.named_parameters(
      ('struct', (np.float16, np.float32)),
      ('nested', (np.float16, [np.float32, (np.float64, [3])])),
  )
  def test_clip_preserves_aggregated_dtype_with_mixed_float(self, type_spec):
    factory = _clipped_sum()
    mixed_float = computation_types.to_type(type_spec)
    process = factory.create(mixed_float)
    type_test_utils.assert_types_identical(
        mixed_float, process.next.type_signature.result.result.member
    )

  @parameterized.named_parameters(
      ('norm_order_1_struct', 1.0, (np.float16, np.float32)),
      ('norm_order_2_struct', 2.0, (np.float16, np.float32)),
      ('norm_order_inf_struct', float('inf'), (np.float16, np.float32)),
      (
          'norm_order_1_nested',
          1.0,
          (np.float16, [np.float32, (np.float64, [3])]),
      ),
      (
          'norm_order_2_nested',
          2.0,
          (np.float16, [np.float32, (np.float64, [3])]),
      ),
      (
          'norm_order_inf_nested',
          float('inf'),
          (np.float16, [np.float32, (np.float64, [3])]),
      ),
  )
  def test_zero_preserves_aggregated_dtype_with_mixed_float(
      self, norm_order, type_spec
  ):
    factory = _zeroed_sum(norm_order=norm_order)
    mixed_float = computation_types.to_type(type_spec)
    process = factory.create(mixed_float)
    type_test_utils.assert_types_identical(
        mixed_float, process.next.type_signature.result.result.member
    )

  @parameterized.named_parameters(
      ('clip_float_on_clients', 1.0, placements.CLIENTS, _clipped_mean),
      ('clip_string_on_server', 'bad', placements.SERVER, _clipped_mean),
      ('zero_float_on_clients', 1.0, placements.CLIENTS, _zeroed_mean),
      ('zero_string_on_server', 'bad', placements.SERVER, _zeroed_mean),
  )
  def test_raises_on_bad_norm_process_result(
      self, value, placement, make_factory
  ):
    report_fn = federated_computation.federated_computation(
        lambda s: intrinsics.federated_value(value, placement), _float_at_server
    )
    norm = _test_norm_process(report_fn=report_fn)

    with self.assertRaisesRegex(TypeError, r'Result type .* assignable to'):
      make_factory(norm)

  @parameterized.named_parameters(
      ('clip', _clipped_mean),
      ('zero', _zeroed_mean),
  )
  def test_raises_on_bad_process_next_single_param(self, make_factory):
    next_fn = federated_computation.federated_computation(
        lambda state: state, _float_at_server
    )
    norm = _test_norm_process(next_fn=next_fn)

    with self.assertRaisesRegex(TypeError, '.* must take two arguments.'):
      make_factory(norm)

  @parameterized.named_parameters(
      ('clip', _clipped_mean),
      ('zero', _zeroed_mean),
  )
  def test_raises_on_bad_process_next_three_params(self, make_factory):
    next_fn = federated_computation.federated_computation(
        lambda state, value1, value2: state,
        _float_at_server,
        _float_at_clients,
        _float_at_clients,
    )
    norm = _test_norm_process(next_fn=next_fn)

    with self.assertRaisesRegex(TypeError, '.* must take two arguments.'):
      make_factory(norm)

  @parameterized.named_parameters(
      ('clip', _clipped_mean),
      ('zero', _zeroed_mean),
  )
  def test_raises_on_bad_process_next_not_float(self, make_factory):
    complex_at_clients = computation_types.FederatedType(
        np.complex64, placements.CLIENTS
    )
    next_fn = federated_computation.federated_computation(
        lambda state, value: state, _float_at_server, complex_at_clients
    )
    norm = _test_norm_process(next_fn=next_fn)

    with self.assertRaisesRegex(
        TypeError, 'Second argument .* assignable from'
    ):
      make_factory(norm)

  @parameterized.named_parameters(
      ('clip', _clipped_mean),
      ('zero', _zeroed_mean),
  )
  def test_raises_on_bad_process_next_two_outputs(self, make_factory):
    next_fn = federated_computation.federated_computation(
        lambda state, val: (state, state), _float_at_server, _float_at_clients
    )
    norm = _test_norm_process(next_fn=next_fn)

    with self.assertRaisesRegex(TypeError, 'Result type .* state only.'):
      make_factory(norm)


class ClippingFactoryExecutionTest(tf.test.TestCase, parameterized.TestCase):

  def _check_result(self, expected, result):
    for exp, res in zip(_make_test_struct_value(expected), result):
      self.assertAllClose(exp, res)

  def test_fixed_clip_sum(self):
    factory = _clipped_sum()

    value_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type)

    state = process.initialize()

    client_data = [1.0, 3.0, 5.0]
    output = process.next(state, client_data)
    self.assertAllClose(5.0, output.result)
    self.assertAllClose(2.0, output.measurements['clipping_norm'])
    self.assertEqual(2, output.measurements['clipped_count'])

  def test_fixed_clip_mean(self):
    factory = _clipped_mean()

    value_type = computation_types.TensorType(np.float32)
    weight_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type, weight_type)

    state = process.initialize()

    client_data = [1.0, 3.0, 5.0]
    client_weight = [1.0, 2.0, 1.0]
    output = process.next(state, client_data, client_weight)
    self.assertAllClose(7 / 4, output.result)
    self.assertAllClose(2.0, output.measurements['clipping_norm'])
    self.assertEqual(2, output.measurements['clipped_count'])

  def test_fixed_clip_sum_struct(self):
    factory = _clipped_sum(4.0)

    value_type = computation_types.to_type(_test_struct_type)
    process = factory.create(value_type)

    state = process.initialize()

    # Struct has 4 components so global norm is twice the constant value.
    client_data = [_make_test_struct_value(v) for v in [1.0, 2.0, 3.0]]
    output = process.next(state, client_data)
    self._check_result(5.0, output.result)
    self.assertAllClose(4.0, output.measurements['clipping_norm'])
    self.assertEqual(1, output.measurements['clipped_count'])

  def test_fixed_clip_mean_struct(self):
    factory = _clipped_mean(4.0)

    value_type = computation_types.to_type(_test_struct_type)
    weight_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type, weight_type)

    state = process.initialize()

    # Struct has 4 components so global norm is twice the constant value.
    client_data = [_make_test_struct_value(v) for v in [1.0, 2.0, 3.0]]
    client_weight = [1.0, 2.0, 1.0]
    output = process.next(state, client_data, client_weight)
    self._check_result(7 / 4, output.result)
    self.assertAllClose(4.0, output.measurements['clipping_norm'])
    self.assertEqual(1, output.measurements['clipped_count'])

  def test_increasing_clip_sum(self):
    factory = _clipped_sum(_test_norm_process())

    value_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type)

    state = process.initialize()

    client_data = [1.0, 3.0, 5.0]
    output = process.next(state, client_data)
    self.assertAllClose(3.0, output.result)
    self.assertAllClose(1.0, output.measurements['clipping_norm'])
    self.assertEqual(2, output.measurements['clipped_count'])

    output = process.next(output.state, client_data)
    self.assertAllClose(5.0, output.result)
    self.assertAllClose(2.0, output.measurements['clipping_norm'])
    self.assertEqual(2, output.measurements['clipped_count'])

    output = process.next(output.state, client_data)
    self.assertAllClose(7.0, output.result)
    self.assertAllClose(3.0, output.measurements['clipping_norm'])
    self.assertEqual(1, output.measurements['clipped_count'])

  def test_increasing_clip_mean(self):
    factory = _clipped_mean(_test_norm_process())

    value_type = computation_types.TensorType(np.float32)
    weight_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type, weight_type)

    state = process.initialize()

    client_data = [1.0, 3.0, 5.0]
    client_weight = [1.0, 2.0, 1.0]
    output = process.next(state, client_data, client_weight)
    self.assertAllClose(1.0, output.result)
    self.assertAllClose(1.0, output.measurements['clipping_norm'])
    self.assertEqual(2, output.measurements['clipped_count'])

    output = process.next(output.state, client_data, client_weight)
    self.assertAllClose(7 / 4, output.result)
    self.assertAllClose(2.0, output.measurements['clipping_norm'])
    self.assertEqual(2, output.measurements['clipped_count'])

    output = process.next(output.state, client_data, client_weight)
    self.assertAllClose(10 / 4, output.result)
    self.assertAllClose(3.0, output.measurements['clipping_norm'])
    self.assertEqual(1, output.measurements['clipped_count'])

  def test_clip_mixed_float_dtype(self):
    factory = _clipped_sum(clip=3.0)
    mixed_float = computation_types.to_type((np.float16, np.float32))
    process = factory.create(mixed_float)

    # Should not clip anything.
    output = process.next(process.initialize(), [(1.0, 1.0), (0.5, 0.5)])
    self.assertAllClose([1.5, 1.5], output.result)

    # Should clip 2nd client contribution by factor of 3/5 (reduced precision).
    output = process.next(process.initialize(), [(1.0, 1.0), (3.0, 4.0)])
    self.assertAllClose([2.8, 3.4], output.result, atol=1e-3, rtol=1e-3)
    output = process.next(process.initialize(), [(1.0, 1.0), (4.0, 3.0)])
    self.assertAllClose([3.4, 2.8], output.result, atol=1e-3, rtol=1e-3)

  def test_fixed_zero_sum(self):
    factory = _zeroed_sum()

    value_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type)

    state = process.initialize()

    client_data = [1.0, 2.0, 5.0]
    output = process.next(state, client_data)
    self.assertAllClose(3.0, output.result)
    self.assertAllClose(2.0, output.measurements['zeroing_norm'])
    self.assertEqual(1, output.measurements['zeroed_count'])

  def test_fixed_zero_mean(self):
    factory = _zeroed_mean()

    value_type = computation_types.TensorType(np.float32)
    weight_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type, weight_type)

    state = process.initialize()

    client_data = [1.0, 2.0, 5.0]
    client_weight = [1.0, 2.0, 2.0]
    output = process.next(state, client_data, client_weight)
    self.assertAllClose(5 / 5, output.result)
    self.assertAllClose(2.0, output.measurements['zeroing_norm'])
    self.assertEqual(1, output.measurements['zeroed_count'])

  def test_fixed_zero_sum_struct(self):
    factory = _zeroed_sum(4.0)

    value_type = computation_types.to_type(_test_struct_type)
    process = factory.create(value_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [1.0, 2.0, 5.0]]
    output = process.next(state, client_data)
    self._check_result(3.0, output.result)
    self.assertAllClose(4.0, output.measurements['zeroing_norm'])
    self.assertEqual(1, output.measurements['zeroed_count'])

  def test_fixed_zero_mean_struct(self):
    factory = _zeroed_mean(4.0)

    value_type = computation_types.to_type(_test_struct_type)
    weight_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type, weight_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [1.0, 2.0, 5.0]]
    client_weight = [1.0, 2.0, 2.0]
    output = process.next(state, client_data, client_weight)
    self._check_result(5 / 5, output.result)
    self.assertAllClose(4.0, output.measurements['zeroing_norm'])
    self.assertEqual(1, output.measurements['zeroed_count'])

  def test_fixed_zero_sum_struct_inf_norm(self):
    factory = _zeroed_sum(2.0, float('inf'))

    value_type = computation_types.to_type(_test_struct_type)
    process = factory.create(value_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [1.0, 2.0, 5.0]]
    output = process.next(state, client_data)
    self._check_result(3.0, output.result)
    self.assertAllClose(2.0, output.measurements['zeroing_norm'])
    self.assertEqual(1, output.measurements['zeroed_count'])

  def test_fixed_zero_mean_struct_inf_norm(self):
    factory = _zeroed_mean(2.0, float('inf'))

    value_type = computation_types.to_type(_test_struct_type)
    weight_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type, weight_type)

    state = process.initialize()

    client_data = [_make_test_struct_value(v) for v in [1.0, 2.0, 5.0]]
    client_weight = [1.0, 2.0, 2.0]
    output = process.next(state, client_data, client_weight)
    self._check_result(5 / 5, output.result)
    self.assertAllClose(2.0, output.measurements['zeroing_norm'])
    self.assertEqual(1, output.measurements['zeroed_count'])

  def test_increasing_zero_sum(self):
    factory = _zeroed_sum(_test_norm_process())

    value_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type)

    state = process.initialize()

    client_data = [0.5, 1.5, 2.5]
    output = process.next(state, client_data)
    self.assertAllClose(0.5, output.result)
    self.assertAllClose(1.0, output.measurements['zeroing_norm'])
    self.assertEqual(2, output.measurements['zeroed_count'])

    output = process.next(output.state, client_data)
    self.assertAllClose(2.0, output.result)
    self.assertAllClose(2.0, output.measurements['zeroing_norm'])
    self.assertEqual(1, output.measurements['zeroed_count'])

    output = process.next(output.state, client_data)
    self.assertAllClose(4.5, output.result)
    self.assertAllClose(3.0, output.measurements['zeroing_norm'])
    self.assertEqual(0, output.measurements['zeroed_count'])

  def test_increasing_zero_mean(self):
    factory = _zeroed_mean(_test_norm_process())

    value_type = computation_types.TensorType(np.float32)
    weight_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type, weight_type)

    state = process.initialize()

    client_data = [0.5, 1.5, 2.5]
    client_weight = [1.0, 2.0, 1.0]
    output = process.next(state, client_data, client_weight)
    self.assertAllClose(0.5 / 4, output.result)
    self.assertAllClose(1.0, output.measurements['zeroing_norm'])
    self.assertEqual(2, output.measurements['zeroed_count'])

    output = process.next(output.state, client_data, client_weight)
    self.assertAllClose(3.5 / 4, output.result)
    self.assertAllClose(2.0, output.measurements['zeroing_norm'])
    self.assertEqual(1, output.measurements['zeroed_count'])

    output = process.next(output.state, client_data, client_weight)
    self.assertAllClose(6 / 4, output.result)
    self.assertAllClose(3.0, output.measurements['zeroing_norm'])
    self.assertEqual(0, output.measurements['zeroed_count'])

  def test_increasing_zero_clip_sum(self):
    # Tests when zeroing and clipping are performed with non-integer clips.
    # Zeroing norm grows by 0.75 each time, clipping norm grows by 0.25.

    @federated_computation.federated_computation(
        _float_at_server, _float_at_clients
    )
    def zeroing_next_fn(state, value):
      del value
      return intrinsics.federated_map(
          tensorflow_computation.tf_computation(lambda x: x + 0.75, np.float32),
          state,
      )

    @federated_computation.federated_computation(
        _float_at_server, _float_at_clients
    )
    def clipping_next_fn(state, value):
      del value
      return intrinsics.federated_map(
          tensorflow_computation.tf_computation(lambda x: x + 0.25, np.float32),
          state,
      )

    zeroing_norm_process = estimation_process.EstimationProcess(
        _test_init_fn, zeroing_next_fn, _test_report_fn
    )
    clipping_norm_process = estimation_process.EstimationProcess(
        _test_init_fn, clipping_next_fn, _test_report_fn
    )

    factory = robust.zeroing_factory(
        zeroing_norm_process, _clipped_sum(clipping_norm_process)
    )

    value_type = computation_types.TensorType(np.float32)
    process = factory.create(value_type)

    state = process.initialize()

    client_data = [1.0, 2.0, 3.0]
    output = process.next(state, client_data)
    self.assertAllClose(1.0, output.measurements['zeroing_norm'])
    self.assertAllClose(1.0, output.measurements['zeroing']['clipping_norm'])
    self.assertEqual(2, output.measurements['zeroed_count'])
    self.assertEqual(0, output.measurements['zeroing']['clipped_count'])
    self.assertAllClose(1.0, output.result)

    output = process.next(output.state, client_data)
    self.assertAllClose(1.75, output.measurements['zeroing_norm'])
    self.assertAllClose(1.25, output.measurements['zeroing']['clipping_norm'])
    self.assertEqual(2, output.measurements['zeroed_count'])
    self.assertEqual(0, output.measurements['zeroing']['clipped_count'])
    self.assertAllClose(1.0, output.result)

    output = process.next(output.state, client_data)
    self.assertAllClose(2.5, output.measurements['zeroing_norm'])
    self.assertAllClose(1.5, output.measurements['zeroing']['clipping_norm'])
    self.assertEqual(1, output.measurements['zeroed_count'])
    self.assertEqual(1, output.measurements['zeroing']['clipped_count'])
    self.assertAllClose(2.5, output.result)

    output = process.next(output.state, client_data)
    self.assertAllClose(3.25, output.measurements['zeroing_norm'])
    self.assertAllClose(1.75, output.measurements['zeroing']['clipping_norm'])
    self.assertEqual(0, output.measurements['zeroed_count'])
    self.assertEqual(2, output.measurements['zeroing']['clipped_count'])
    self.assertAllClose(4.5, output.result)

    output = process.next(output.state, client_data)
    self.assertAllClose(4.0, output.measurements['zeroing_norm'])
    self.assertAllClose(2.0, output.measurements['zeroing']['clipping_norm'])
    self.assertEqual(0, output.measurements['zeroed_count'])
    self.assertEqual(1, output.measurements['zeroing']['clipped_count'])
    self.assertAllClose(5.0, output.result)

  @parameterized.named_parameters(
      ('norm_order_1', 1.0),
      ('norm_order_2', 2.0),
      ('norm_order_inf', float('inf')),
  )
  def test_zero_mixed_float_dtype(self, norm_order):
    factory = _zeroed_sum(clip=3.0, norm_order=norm_order)
    mixed_float = computation_types.to_type((np.float16, np.float32))
    process = factory.create(mixed_float)

    # Should not zero-out anything.
    output = process.next(process.initialize(), [(1.0, 1.0), (0.5, 0.5)])
    self.assertAllClose([1.5, 1.5], output.result)

    # Should zero-out 2nd client contribution.
    output = process.next(process.initialize(), [(1.0, 1.0), (10.0, 1.0)])
    self.assertAllClose([1.0, 1.0], output.result)
    output = process.next(process.initialize(), [(1.0, 1.0), (1.0, 10.0)])
    self.assertAllClose([1.0, 1.0], output.result)
    output = process.next(process.initialize(), [(1.0, 1.0), (10.0, 10.0)])
    self.assertAllClose([1.0, 1.0], output.result)


class NormTest(tf.test.TestCase):

  def test_norms(self):
    values = [1.0, -2.0, 2.0, -4.0]
    for l in itertools.permutations(values):
      v = [tf.constant(l[0]), (tf.constant([l[1], l[2]]), tf.constant([l[3]]))]
      self.assertAllClose(4.0, robust._global_inf_norm(v).numpy())
      self.assertAllClose(5.0, robust._global_l2_norm(v).numpy())
      self.assertAllClose(9.0, robust._global_l1_norm(v).numpy())

  def test_clip_by_global_l2_norm(self):
    """Test `_clip_by_global_l2_norm` equivalent to `tf.clip_by_global_norm`."""
    value = [
        tf.constant([1.2]),
        tf.constant([3.5, -9.1]),
        tf.constant([[1.1, -1.1], [0.0, 4.2]]),
    ]
    for clip_norm in [0.0, 1.0, 100.0]:
      self.assertAllClose(
          robust._clip_by_global_l2_norm(value, clip_norm),
          tf.clip_by_global_norm(value, clip_norm),
      )


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
