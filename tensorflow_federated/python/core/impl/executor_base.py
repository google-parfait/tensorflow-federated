# Copyright 2018, The TensorFlow Federated Authors.
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
"""An abstract base interface to be implemented by executors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

# Dependency imports
import six


@six.add_metaclass(abc.ABCMeta)
class Executor(object):
  """The abstract base interface to be implemented by executors."""

  @abc.abstractmethod
  def execute(self, computation_proto):
    """Executes `computation_proto` and returns the final results.

    Args:
      computation_proto: A self-contained instance of the `Computation` proto,
        i.e., one that does not declare any parameters.

    Returns:
      The result produced by the computation (the format of which depends on
      the type of the executor supported).
    """
    raise NotImplementedError
