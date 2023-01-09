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
"""Asbtract interface for objects that carry information about cardinality."""

import abc


class CardinalityCarrying(metaclass=abc.ABCMeta):
  """Asbtract interface for objects that carry information about cardinality."""

  @property
  @abc.abstractmethod
  def cardinality(self):
    """Returns the cardinality information associated with this object.

    Returns:
      A dictionary in which the keys are placements, and values are integers
      that indicate the cardinalities associated with these placements. The
      cardinality payload might contain cardinalities for zero, one, or more
      than one placement.
    """
    raise NotImplementedError
