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
"""Utilities for testing aggregation factories and processes."""

import typing

import federated_language
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

MEASUREMENT_CONSTANT = 42


class SumPlusOneFactory(factory.UnweightedAggregationFactory):
  """`UnweightedAggregationFactory` for "sum + 1".

  The created `tff.templates.AggregationProcess` will sum the values placed at
  `CLIENTS`, and add `1` to the sum. In the case of value of structured type,
  `1` will be added to each element. The `state` of the process is initialized
  as `0` and incremented by `1` in each iteration of `next`. The `measurements`
  are always equal to `42`.

  This factory is intended as a testing utility for testing aggregation
  factories which are optionally be parameterized by inner aggregation
  factories. Due to the relative simplicity of this specific aggregation, it
  should be easy to verify whether it was actually applied, when testing an
  outer aggregation factory.
  """

  def create(
      self, value_type: factory.ValueType
  ) -> aggregation_process.AggregationProcess:
    type_args = typing.get_args(factory.ValueType)
    py_typecheck.check_type(value_type, type_args)

    @federated_language.federated_computation()
    def init_fn():
      return federated_language.federated_value(0, federated_language.SERVER)

    @federated_language.federated_computation(
        init_fn.type_signature.result,
        federated_language.FederatedType(
            value_type, federated_language.CLIENTS
        ),
    )
    def next_fn(state, value):
      state = federated_language.federated_map(
          tensorflow_computation.tf_computation(lambda x: x + 1), state
      )
      result = federated_language.federated_map(
          tensorflow_computation.tf_computation(
              lambda x: tf.nest.map_structure(lambda y: y + 1, x)
          ),
          federated_language.federated_sum(value),
      )
      measurements = federated_language.federated_value(
          MEASUREMENT_CONSTANT, federated_language.SERVER
      )
      return measured_process.MeasuredProcessOutput(state, result, measurements)

    return aggregation_process.AggregationProcess(init_fn, next_fn)
