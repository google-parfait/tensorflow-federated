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

_float_type = computation_types.to_type(tf.float32)
_float_type_clients = computation_types.at_clients(_float_type)


def _get_min(value):
  min_value = primitives.federated_min(value)
  return collections.OrderedDict(min_value=min_value)


@computations.tf_computation
def _mul(value, weight):
  return value * weight


def _get_weighted_min(value, weight):
  weighted_value = intrinsics.federated_map(_mul, (value, weight))
  min_weighted_value = primitives.federated_min(weighted_value)
  return collections.OrderedDict(min_weighted_value=min_weighted_value)


_struct_type = computation_types.to_type([(tf.float32, (3,)), tf.float32])
_struct_type_clients = computation_types.at_clients(_struct_type)


@computations.tf_computation
def _get_norm(value):
  return tf.linalg.global_norm(tf.nest.flatten(value))


def _make_struct(x):
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
          unweighted_factory, _get_weighted_min, client_placed_input=True)

    with self.assertRaisesRegex(ValueError, 'single parameter'):
      measurements.add_measurements(
          unweighted_factory, _get_weighted_min, client_placed_input=False)

    weighted_factory = mean.MeanFactory()
    with self.assertRaisesRegex(ValueError, 'two parameters'):
      measurements.add_measurements(
          weighted_factory, _get_min, client_placed_input=True)

  def test_unweighted(self):
    factory = sum_factory.SumFactory()
    factory = measurements.add_measurements(
        factory, _get_min, client_placed_input=True)
    process = factory.create(_float_type)

    state = process.initialize()
    client_data = [1.0, 2.0, 3.0]
    output = process.next(state, client_data)
    self.assertAllClose(6.0, output.result)
    self.assertDictEqual(
        collections.OrderedDict(min_value=1.0), output.measurements)

  def test_weighted(self):
    factory = mean.MeanFactory()
    factory = measurements.add_measurements(
        factory, _get_weighted_min, client_placed_input=True)
    process = factory.create(_float_type, _float_type)

    state = process.initialize()
    client_values = [1.0, 2.0, 3.0]
    client_weights = [3.0, 1.0, 2.0]
    output = process.next(state, client_values, client_weights)
    self.assertAllClose(11 / 6, output.result)
    self.assertDictEqual(
        collections.OrderedDict(
            mean_value=(), mean_weight=(), min_weighted_value=2.0),
        output.measurements)

  def test_unweighted_struct(self):
    factory = sum_factory.SumFactory()
    factory = measurements.add_measurements(
        factory, _get_min_norm, client_placed_input=True)
    process = factory.create(_struct_type)

    state = process.initialize()
    client_data = [_make_struct(x) for x in [1.0, 2.0, 3.0]]
    output = process.next(state, client_data)
    self.assertAllClose(_make_struct(6.0), output.result)
    self.assertDictEqual(
        collections.OrderedDict(min_norm=2.0), output.measurements)

  def test_weighted_struct(self):
    factory = mean.MeanFactory()
    factory = measurements.add_measurements(
        factory, _get_min_weighted_norm, client_placed_input=True)
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

  def test_unweighted_server_measurements(self):
    factory = sum_factory.SumFactory()
    factory = measurements.add_measurements(
        factory, _get_server_norm, client_placed_input=False)
    process = factory.create(_struct_type)

    state = process.initialize()
    client_data = [_make_struct(x) for x in [1.0, 2.0, 3.0]]
    output = process.next(state, client_data)
    self.assertAllClose(_make_struct(6.0), output.result)
    self.assertDictEqual(
        collections.OrderedDict(server_norm=12.0), output.measurements)

  def test_weighted_server_measurements(self):
    factory = mean.MeanFactory()
    factory = measurements.add_measurements(
        factory, _get_server_norm, client_placed_input=False)
    process = factory.create(_struct_type, _float_type)

    state = process.initialize()
    client_data = [_make_struct(x) for x in [1.0, 2.0, 3.0]]
    client_weights = [3.0, 1.0, 2.0]
    output = process.next(state, client_data, client_weights)
    self.assertAllClose(_make_struct(11 / 6), output.result)
    self.assertAllClose(
        collections.OrderedDict(
            mean_value=(), mean_weight=(), server_norm=11 / 3),
        output.measurements)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
