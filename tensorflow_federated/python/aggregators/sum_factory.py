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
"""Factory for summation."""

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class SumFactory(factory.UnweightedAggregationFactory):
  """`UnweightedAggregationFactory` for sum.

  The created `tff.templates.AggregationProcess` sums values placed at
  `CLIENTS`, and outputs the sum placed at `SERVER`.

  The process has empty `state` and returns no `measurements`. For summation,
  implementation delegates to the `tff.federated_sum` operator.
  """

  def create(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:

    py_typecheck.check_type(value_type, factory.ValueType.__args__)

    @computations.federated_computation()
    def init_fn():
      return intrinsics.federated_value((), placements.SERVER)

    @computations.federated_computation(init_fn.type_signature.result,
                                        computation_types.FederatedType(
                                            value_type, placements.CLIENTS))
    def next_fn(state, value):
      summed_value = intrinsics.federated_sum(value)
      empty_measurements = intrinsics.federated_value((), placements.SERVER)
      return measured_process.MeasuredProcessOutput(state, summed_value,
                                                    empty_measurements)

    return aggregation_process.AggregationProcess(init_fn, next_fn)
