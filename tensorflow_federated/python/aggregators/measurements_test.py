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
"""Tests for factory_with_additional_measurements."""

import collections

import tensorflow as tf

from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import measurements
from tensorflow_federated.python.aggregators import primitives
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types

_struct_type = computation_types.to_type([(tf.float32, (3,)), tf.float32])
_struct_type_clients = computation_types.at_clients(_struct_type)
_float_type = computation_types.to_type(tf.float32)


@computations.tf_computation
def _get_norm(value):
  return tf.linalg.global_norm(tf.nest.flatten(value))


def _make_struct(x):
  # L2 norm of this struct is just 2 * x.
  return [tf.constant(x, dtype=tf.float32, shape=(3,)), x]


def _get_min_norm(value):
  norms = intrinsics.federated_map(_get_norm, value)
  min_norm = primitives.federated_min(norms)
  return collections.OrderedDict(min_norm=min_norm)


@computations.tf_computation
def _mul_struct(value, weight):
  return tf.nest.map_structure(lambda x: x * weight, value)


def _get_min_weighted_norm(value, weight):
  weighted_value = intrinsics.federated_map(_mul_struct, (value, weight))
  norms = intrinsics.federated_map(_get_norm, weighted_value)
  min_weighted_norm = primitives.federated_min(norms)
  return collections.OrderedDict(min_weighted_norm=min_weighted_norm)


def _get_server_norm(value):
  server_norm = intrinsics.federated_map(_get_norm, value)
  return collections.OrderedDict(server_norm=server_norm)


class AddMeasurementsTest(test_case.TestCase):

  def test_raises_bad_measurement_fn(self):
    unweighted_factory = sum_factory.SumFactory()
    with self.assertRaisesRegex(ValueError, 'single parameter'):
      measurements.add_measurements(
          unweighted_factory, client_measurement_fn=_get_min_weighted_norm)

    with self.assertRaisesRegex(ValueError, 'single parameter'):
      measurements.add_measurements(
          unweighted_factory, server_measurement_fn=_get_min_weighted_norm)

    weighted_factory = mean.MeanFactory()
    with self.assertRaisesRegex(ValueError, 'two parameters'):
      measurements.add_measurements(
          weighted_factory, client_measurement_fn=_get_min_norm)

  def test_unweighted_client(self):
    factory = sum_factory.SumFactory()

    factory = measurements.add_measurements(
        factory, client_measurement_fn=_get_min_norm)
    process = factory.create(_struct_type)

    state = process.initialize()
    client_data = [_make_struct(x) for x in [1.0, 2.0, 3.0]]
    output = process.next(state, client_data)
    self.assertAllClose(_make_struct(6.0), output.result)
    self.assertDictEqual(
        collections.OrderedDict(min_norm=2.0), output.measurements)

  def test_weighted_client(self):
    factory = mean.MeanFactory()

    factory = measurements.add_measurements(
        factory, client_measurement_fn=_get_min_weighted_norm)
    process = factory.create(_struct_type, _float_type)

    state = process.initialize()
    client_data = [_make_struct(x) for x in [1.0, 2.0, 3.0]]
    client_weights = [3.0, 1.0, 2.0]
    output = process.next(state, client_data, client_weights)
    self.assertAllClose(_make_struct(11 / 6), output.result)
    self.assertDictEqual(
        collections.OrderedDict(
            mean_value=(), mean_weight=(), min_weighted_norm=4.0),
        output.measurements)

  def test_server(self):
    factory = sum_factory.SumFactory()

    factory = measurements.add_measurements(
        factory, server_measurement_fn=_get_server_norm)
    process = factory.create(_struct_type)

    state = process.initialize()
    client_data = [_make_struct(x) for x in [1.0, 2.0, 3.0]]
    output = process.next(state, client_data)
    self.assertAllClose(_make_struct(6.0), output.result)
    self.assertDictEqual(
        collections.OrderedDict(server_norm=12.0), output.measurements)

  def test_unweighted_client_and_server(self):
    factory = sum_factory.SumFactory()

    factory = measurements.add_measurements(
        factory,
        client_measurement_fn=_get_min_norm,
        server_measurement_fn=_get_server_norm)
    process = factory.create(_struct_type)

    state = process.initialize()
    client_data = [_make_struct(x) for x in [1.0, 2.0, 3.0]]
    output = process.next(state, client_data)
    self.assertAllClose(_make_struct(6.0), output.result)
    self.assertDictEqual(
        collections.OrderedDict(min_norm=2.0, server_norm=12.0),
        output.measurements)

  def test_weighted_client_and_server(self):
    factory = mean.MeanFactory()

    factory = measurements.add_measurements(
        factory,
        client_measurement_fn=_get_min_weighted_norm,
        server_measurement_fn=_get_server_norm)
    process = factory.create(_struct_type, _float_type)

    state = process.initialize()
    client_data = [_make_struct(x) for x in [1.0, 2.0, 3.0]]
    client_weights = [3.0, 1.0, 2.0]
    output = process.next(state, client_data, client_weights)
    self.assertAllClose(_make_struct(11 / 6), output.result)
    self.assertAllClose(
        collections.OrderedDict(
            mean_value=(),
            mean_weight=(),
            min_weighted_norm=4.0,
            server_norm=11 / 3), output.measurements)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
