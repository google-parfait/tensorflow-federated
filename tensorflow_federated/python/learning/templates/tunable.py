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
"""Abstractions for tunable algorithms."""

import abc
from typing import Any, Callable, Generic, OrderedDict, TypeVar

Hparams = OrderedDict[str, Any]
State = TypeVar('State')


class Tunable(abc.ABC, Generic[State]):
  """Represents a tunable algorithms for use in TensorFlow Federated.

  This abstract interface defines `get_hparams` and `set_hparams` methods that
  are used, respectively, to extract hyperparameters from some state, and to set
  the hyperparameters of that state. This is intended to allow users to tune
  hyperparameters of the algorithm.

  These methods are intended to be used in settings where an algorithm is
  tunable, but the state object is complex or should not be altered directly.
  """

  @property
  @abc.abstractmethod
  def get_hparams(self) -> Callable[[State], Hparams]:
    """A `tff.Computation` that gets the hyperparameters of an algorithm state.

    This computation accepts the (unplaced) state of an algorithm, and returns
    its associated hyperparameters.

    Returns:
      A `tff.Computation`.
    """

  @property
  @abc.abstractmethod
  def set_hparams(self) -> Callable[[State, Hparams], State]:
    """A `tff.Computation` that sets the hyperparameters of an algorithm state.

    This computation accepts the (unplaced) state of an algorithm and a
    structure of hyperparameters (matching the output of the `get_hparams`
    computation) and returning some updated state.

    Returns:
      A `tff.Computation`.
    """
