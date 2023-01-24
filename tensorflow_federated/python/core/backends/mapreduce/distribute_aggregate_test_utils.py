# Copyright 2022, The TensorFlow Federated Authors.
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
"""Utilities for testing the DistributeAggregateForm backend."""

from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.types import computation_types


def generate_unnamed_type_signature(
    client_work: computation_impl.ConcreteComputation,
    server_result: computation_impl.ConcreteComputation,
) -> computation_types.FunctionType:
  """Generates a type signature for the DistributeAggregateForm."""
  parameter = computation_types.StructType([
      server_result.type_signature.parameter[0],
      client_work.type_signature.parameter[0],
  ])
  result = computation_types.StructType([
      server_result.type_signature.parameter[0],
      server_result.type_signature.result[1],
  ])
  return computation_types.FunctionType(parameter, result)
