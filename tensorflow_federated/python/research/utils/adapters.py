# Lint as: python3
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
"""Adapters to map between TFF constructs and python containers."""

import abc

import attr


class IterativeProcessPythonAdapter(metaclass=abc.ABCMeta):
  """Converts iterative process results from anonymous tuples."""

  @abc.abstractmethod
  def initialize(self):
    """Returns the initial state of the iterative process.

    Returns:
      The initial state of the iterative process, after converting from
      anonymous tuple.
    """
    pass

  @abc.abstractmethod
  def next(self, state, data):
    """Performs a single step of iteration.

    Args:
      state: The state of the iterative process.
      data: The next set of data to use for iteration.

    Returns:
      An IterationResult containing the result of one step of the iterative
        process.
    """
    pass


@attr.s(eq=False)
class IterationResult(object):
  """Holds the results of an iteration: the state, metrics and output."""
  state = attr.ib()
  metrics = attr.ib()
  output = attr.ib()
