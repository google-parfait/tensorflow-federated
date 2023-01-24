# Copyright 2019, The TensorFlow Federated Authors.
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
"""ExecutorFactory interface and simple implementation."""

import abc
from collections.abc import MutableMapping

from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.types import placements

CardinalitiesType = MutableMapping[placements.PlacementLiteral, int]


class ExecutorFactory(metaclass=abc.ABCMeta):
  """Interface defining executor factories.

  `ExecutorFactory` should be considered to own the executors it creates; it
  is responsible for their instantiation and management.

  `ExecutorFactory` exposes two methods, `create_executor` and
  `clean_up_executors`. There is a particular coupling between these two
  methods; any executor returned by `create_executor` should not be used
  after `clean_up_executors` has been called without reinitialization. That is,
  `create_executor` should be called again, and `ExecutorFactory` will ensure
  that the returned executor is safe for use.
  """

  @abc.abstractmethod
  def create_executor(
      self, cardinalities: CardinalitiesType
  ) -> executor_base.Executor:
    """Abstract method to construct instance of `executor_base.Executor`.

    `create_executor` must accept a dict mapping
    `placements.PlacementLiterals` to `ints`, and return an
    `executor_base.Executor`.

    Args:
      cardinalities: a dict mapping instances of `placements.PlacementLiteral`
        to ints, specifying the population size at each placement.

    Returns:
      Instance of `executor_base.Executor`.
    """
    pass

  @abc.abstractmethod
  def clean_up_executor(self, cardinalities: CardinalitiesType):
    """Releases any resources associated to the given cardinalities.

    Note that calling this method may invalidate the state of any executors
    which have previously been returned by the factory with the `cardinalities`
    argument ; `create_executor` should be called again if a new executor which
    is safe to use is desired.

    Args:
      cardinalities: The cardinalities of the executor whose state we wish to
        clear.
    """
    pass
