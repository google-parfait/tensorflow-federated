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
"""A set of utility methods for serializing Value protos using pybind11 bindings."""

from collections.abc import Mapping, Sequence

import federated_language
from federated_language.proto import computation_pb2

from tensorflow_federated.proto.v0 import executor_pb2


def serialize_cardinalities(
    cardinalities: Mapping[federated_language.framework.PlacementLiteral, int],
) -> list[executor_pb2.Cardinality]:
  serialized_cardinalities = []
  for placement, cardinality in cardinalities.items():
    cardinality_message = executor_pb2.Cardinality(
        placement=computation_pb2.Placement(uri=placement.uri),
        cardinality=cardinality,
    )
    serialized_cardinalities.append(cardinality_message)
  return serialized_cardinalities


def deserialize_cardinalities(
    serialized_cardinalities: Sequence[executor_pb2.Cardinality],
) -> dict[federated_language.framework.PlacementLiteral, int]:
  cardinalities = {}
  for cardinality_spec in serialized_cardinalities:
    literal = federated_language.framework.uri_to_placement_literal(
        cardinality_spec.placement.uri
    )
    cardinalities[literal] = cardinality_spec.cardinality
  return cardinalities
