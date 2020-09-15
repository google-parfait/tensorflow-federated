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
"""Abstract base factory class for creation of `AggregationProcess`."""

import abc
from typing import Union

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.templates import aggregation_process

ValueType = Union[computation_types.TensorType, computation_types.StructType]


class AggregationProcessFactory(abc.ABC):
  """Factory for `tff.templates.AggregationProcess`."""

  @abc.abstractmethod
  def create(self,
             value_type: ValueType) -> aggregation_process.AggregationProcess:
    """Creates a `tff.aggregators.AggregationProcess` aggregating `value_type`.

    The provided `value_type` is a non-federated `tff.Type` object, that is,
    `value_type.is_federated()` should return `False`. Provided `value_type`
    must be a `tff.TensorType` or a `tff.StructType`.

    The returned `tff.aggregators.AggregationProcess` will be created for
    aggregation of values matching `value_type`. That is, its `next` method will
    expect type `<S@SERVER, {value_type}@CLIENTS, *>`, where `S` is the unplaced
    return type of its `initialize` method, and * stands for optional additional
    placed input arguments.

    Args:
      value_type: A `tff.Type` without placement.

    Returns:
      A `tff.templates.AggregationProcess`.
    """
