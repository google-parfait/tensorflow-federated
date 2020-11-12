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

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

MEASUREMENT_CONSTANT = 42


class SumPlusOneFactory(factory.UnweightedAggregationFactory):
  """`NonWeightedAggregationFactory` for "sum + 1".

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

  def create_unweighted(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:

    py_typecheck.check_type(value_type, factory.ValueType.__args__)

    @computations.federated_computation()
    def init_fn():
      return intrinsics.federated_value(0, placements.SERVER)

    @computations.federated_computation(init_fn.type_signature.result,
                                        computation_types.FederatedType(
                                            value_type, placements.CLIENTS))
    def next_fn(state, value):
      state = intrinsics.federated_map(
          computations.tf_computation(lambda x: x + 1), state)
      result = intrinsics.federated_map(
          computations.tf_computation(
              lambda x: tf.nest.map_structure(lambda y: y + 1, x)),
          intrinsics.federated_sum(value))
      measurements = intrinsics.federated_value(MEASUREMENT_CONSTANT,
                                                placements.SERVER)
      return measured_process.MeasuredProcessOutput(state, result, measurements)

    return aggregation_process.AggregationProcess(init_fn, next_fn)
