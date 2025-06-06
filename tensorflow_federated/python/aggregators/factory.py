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
"""Abstract base factory classes for creation of `AggregationProcess`."""

import abc
from typing import Union

import federated_language

from tensorflow_federated.python.core.templates import aggregation_process

ValueType = Union[federated_language.TensorType, federated_language.StructType]


class UnweightedAggregationFactory(abc.ABC):
  """Factory for creating `tff.templates.AggregationProcess` without weights."""

  @abc.abstractmethod
  def create(
      self, value_type: ValueType
  ) -> aggregation_process.AggregationProcess:
    """Creates a `tff.aggregators.AggregationProcess` without weights.

    The provided `value_type` is a non-federated `federated_language.Type`, that
    is, not a
    `federated_language.FederatedType`.

    The returned `tff.aggregators.AggregationProcess` will be created for
    aggregation of values matching `value_type` placed at
    `federated_language.CLIENTS`.
    That is, its `next` method will expect type
    `<S@SERVER, {value_type}@CLIENTS>`, where `S` is the unplaced return type of
    its `initialize` method.

    Args:
      value_type: A non-federated `federated_language.Type`.

    Returns:
      A `tff.templates.AggregationProcess`.
    """
    raise NotImplementedError


class WeightedAggregationFactory(abc.ABC):
  """Factory for creating `tff.templates.AggregationProcess` with weights."""

  @abc.abstractmethod
  def create(
      self, value_type: ValueType, weight_type: ValueType
  ) -> aggregation_process.AggregationProcess:
    """Creates a `tff.aggregators.AggregationProcess` with weights.

    The provided `value_type` and `weight_type` are non-federated
    `federated_language.Type`s.
    That is, neither is a `federated_language.FederatedType`.

    The returned `tff.aggregators.AggregationProcess` will be created
    for aggregation of pairs of values matching `value_type` and `weight_type`
    placed at `federated_language.CLIENTS`. That is, its `next` method will
    expect type
    `<S@SERVER, {value_type}@CLIENTS, {weight_type}@CLIENTS>`, where `S` is the
    unplaced return type of its `initialize` method.

    Args:
      value_type: A non-federated `federated_language.Type`.
      weight_type: A non-federated `federated_language.Type`.

    Returns:
      A `tff.templates.AggregationProcess`.
    """
    raise NotImplementedError


AggregationFactory = Union[
    UnweightedAggregationFactory, WeightedAggregationFactory
]
