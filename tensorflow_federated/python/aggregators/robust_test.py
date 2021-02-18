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
"""Tests for clipping and zeroing robust_factory."""

import collections
import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.aggregators import mean_factory
from tensorflow_federated.python.aggregators import robust_factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process

_test_struct_type = [(tf.float32, (3,)), tf.float32]


def _make_test_struct_value(x):
  return [tf.constant(x, dtype=tf.float32, shape=(3,)), x]


def _clipped_mean(clip=2.0):
  return robust_factory.clipping_factory(clip, mean_factory.MeanFactory())


def _clipped_sum(clip=2.0):
  return robust_factory.clipping_factory(clip, sum_factory.SumFactory())


def _zeroed_mean(clip=2.0, norm_order=2.0):
  return robust_factory.zeroing_factory(clip, mean_factory.MeanFactory(),
                                        norm_order)


def _zeroed_sum(clip=2.0, norm_order=2.0):
  return robust_factory.zeroing_factory(clip, sum_factory.SumFactory(),
                                        norm_order)


_float_at_server = computation_types.at_server(tf.float32)
_float_at_clients = computation_types.at_clients(tf.float32)


@computations.federated_computation()
def _test_init_fn():
  return intrinsics.federated_value(1., placements.SERVER)


@computations.federated_computation(_float_at_server, _float_at_clients)
def _test_next_fn(state, value):
  del value
  return intrinsics.federated_map(
      computations.tf_computation(lambda x: x + 1., tf.float32), state)


@computations.federated_computation(_float_at_server)
def _test_report_fn(state):
  return state


def _test_norm_process(init_fn=_test_init_fn,
                       next_fn=_test_next_fn,
                       report_fn=_test_report_fn):
  return estimation_process.EstimationProcess(init_fn, next_fn, report_fn)


class ClippingFactoryComputationTest(test_case.TestCase,
                                     parameterized.TestCase):

  @parameterized.named_parameters(
      ('float', tf.float32),
      ('struct', _test_struct_type),
  )
  def test_clip_type_properties_simple(self, value_type):
    factory = _clipped_sum()
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    server_state_type = computation_types.at_server(
        collections.OrderedDict(
            clipping_norm=(), inner_agg=(), clipped_count_agg=()))
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(
            clipping=(),
            clipping_norm=robust_factory.NORM_TF_TYPE,
            clipped_count=robust_factory.COUNT_TF_TYPE))
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  @parameterized.named_parameters(
      ('float_value_float32_weight', tf.float32, tf.float32),
      ('struct_value_float32_weight', _test_struct_type, tf.float32),
      ('float_value_float64_weight', tf.float32, tf.float64),
      ('struct_value_float64_weight', _test_struct_type, tf.float64),
  )
  def test_clip_type_properties_weighted(self, value_type, weight_type):
    factory = _clipped_mean()
    value_type = computation_types.to_type(value_type)
    weight_type = computation_types.to_type(weight_type)
    process = factory.create(value_type, weight_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    mean_state_type = collections.OrderedDict(
        value_sum_process=(), weight_sum_process=())
    server_state_type = computation_types.at_server(
        collections.OrderedDict(
            clipping_norm=(), inner_agg=mean_state_type, clipped_count_agg=()))
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(
            clipping=collections.OrderedDict(mean_value=(), mean_weight=()),
            clipping_norm=robust_factory.NORM_TF_TYPE,
            clipped_count=robust_factory.COUNT_TF_TYPE))
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type),
            weight=computation_types.at_clients(weight_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  @parameterized.named_parameters(
      ('float', tf.float32),
      ('struct', _test_struct_type),
  )
  def test_zero_type_properties_simple(self, value_type):
    factory = _zeroed_sum()
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    server_state_type = computation_types.at_server(
        collections.OrderedDict(
            zeroing_norm=(), inner_agg=(), zeroed_count_agg=()))
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(
            zeroing=(),
            zeroing_norm=robust_factory.NORM_TF_TYPE,
            zeroed_count=robust_factory.COUNT_TF_TYPE))
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  @parameterized.named_parameters(
      ('float_value_float32_weight', tf.float32, tf.float32),
      ('struct_value_float32_weight', _test_struct_type, tf.float32),
      ('float_value_float64_weight', tf.float32, tf.float64),
      ('struct_value_float64_weight', _test_struct_type, tf.float64),
  )
  def test_zero_type_properties_weighted(self, value_type, weight_type):
    factory = _zeroed_mean()
    value_type = computation_types.to_type(value_type)
    weight_type = computation_types.to_type(weight_type)
    process = factory.create(value_type, weight_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    mean_state_type = collections.OrderedDict(
        value_sum_process=(), weight_sum_process=())
    server_state_type = computation_types.at_server(
        collections.OrderedDict(
            zeroing_norm=(), inner_agg=mean_state_type, zeroed_count_agg=()))
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(
            zeroing=collections.OrderedDict(mean_value=(), mean_weight=()),
            zeroing_norm=robust_factory.NORM_TF_TYPE,
            zeroed_count=robust_factory.COUNT_TF_TYPE))
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type),
            weight=computation_types.at_clients(weight_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  @parameterized.named_parameters(
      ('clip_float_on_clients', 1.0, placements.CLIENTS, _clipped_mean),
      ('clip_string_on_server', 'bad', placements.SERVER, _clipped_mean),
      ('zero_float_on_clients', 1.0, placements.CLIENTS, _zeroed_mean),
      ('zero_string_on_server', 'bad', placements.SERVER, _zeroed_mean),
  )
  def test_raises_on_bad_norm_process_result(self, value, placement,
                                             make_factory):
    report_fn = computations.federated_computation(
        lambda s: intrinsics.federated_value(value, placement),
        _float_at_server)
    norm = _test_norm_process(report_fn=report_fn)

    with self.assertRaisesRegex(TypeError, r'Result type .* assignable to'):
      make_factory(norm)

  @parameterized.named_parameters(
      ('clip', _clipped_mean),
      ('zero', _zeroed_mean),
  )
  def test_raises_on_bad_process_next_single_param(self, make_factory):
    next_fn = computations.federated_computation(lambda state: state,
                                                 _float_at_server)
    norm = _test_norm_process(next_fn=next_fn)

    with self.assertRaisesRegex(TypeError, '.* must take two arguments.'):
      make_factory(norm)

  @parameterized.named_parameters(
      ('clip', _clipped_mean),
      ('zero', _zeroed_mean),
  )
  def test_raises_on_bad_process_next_three_params(self, make_factory):
    next_fn = computations.federated_computation(
        lambda state, value1, value2: state, _float_at_server,
        _float_at_clients, _float_at_clients)
    norm = _test_norm_process(next_fn=next_fn)

    with self.assertRaisesRegex(TypeError, '.* must take two arguments.'):
      make_factory(norm)

  @parameterized.named_parameters(
      ('clip', _clipped_mean),
      ('zero', _zeroed_mean),
  )
  def test_raises_on_bad_process_next_not_float(self, make_factory):
    complex_at_clients = computation_types.at_clients(tf.complex64)
    next_fn = computations.federated_computation(lambda state, value: state,
                                                 _float_at_server,
                                                 complex_at_clients)
    norm = _test_norm_process(next_fn=next_fn)

    with self.assertRaisesRegex(TypeError,
                                'Second argument .* assignable from'):
      make_factory(norm)

  @parameterized.named_parameters(
      ('clip', _clipped_mean),
      ('zero', _zeroed_mean),
  )
  def test_raises_on_bad_process_next_two_outputs(self, make_factory):
    next_fn = computations.federated_computation(
        lambda state, val: (state, state), _float_at_server, _float_at_clients)
    norm = _test_norm_process(next_fn=next_fn)

    with self.assertRaisesRegex(TypeError, 'Result type .* state only.'):
      make_factory(norm)


class ClippingFactoryExecutionTest(test_case.TestCase):

  def _check_result(self, expected, result):
    for exp, res in zip(_make_test_struct_value(expected), result):
      self.assertAllClose(exp, res)

  def test_fixed_clip_sum(self):
    factory = _clipped_sum()

    value_type = computation_types.to_type(tf.float32)
    process = factory.create(value_type)

    state = process.initialize()

    client_data = [1.0, 3.0, 5.0]
    output = process.next(state, client_data)
    self.assertAllClose(5.0, output.result)
    self.assertAllClose(2.0, output.measurements['clipping_norm'])
    self.assertEqual(2, output.measurements['clipped_count'])

  def test_fixed_clip_mean(self):
    factory = _clipped_mean()

    value_type = computation_types.to_type(tf.float32)
    weight_type = computation_types.to_type(tf.float32)
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
    weight_type = computation_types.to_type(tf.float32)
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

    value_type = computation_types.to_type(tf.float32)
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

    value_type = computation_types.to_type(tf.float32)
    weight_type = computation_types.to_type(tf.float32)
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

  def test_fixed_zero_sum(self):
    factory = _zeroed_sum()

    value_type = computation_types.to_type(tf.float32)
    process = factory.create(value_type)

    state = process.initialize()

    client_data = [1.0, 2.0, 5.0]
    output = process.next(state, client_data)
    self.assertAllClose(3.0, output.result)
    self.assertAllClose(2.0, output.measurements['zeroing_norm'])
    self.assertEqual(1, output.measurements['zeroed_count'])

  def test_fixed_zero_mean(self):
    factory = _zeroed_mean()

    value_type = computation_types.to_type(tf.float32)
    weight_type = computation_types.to_type(tf.float32)
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
    weight_type = computation_types.to_type(tf.float32)
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
    weight_type = computation_types.to_type(tf.float32)
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

    value_type = computation_types.to_type(tf.float32)
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

    value_type = computation_types.to_type(tf.float32)
    weight_type = computation_types.to_type(tf.float32)
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


class NormTest(test_case.TestCase):

  def test_norms(self):
    values = [1.0, -2.0, 3.0, -4.0]
    for l in itertools.permutations(values):
      v = [tf.constant(l[0]), (tf.constant([l[1], l[2]]), tf.constant([l[3]]))]
      self.assertAllClose(4.0, robust_factory._global_inf_norm(v).numpy())
      self.assertAllClose(10.0, robust_factory._global_l1_norm(v).numpy())


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
