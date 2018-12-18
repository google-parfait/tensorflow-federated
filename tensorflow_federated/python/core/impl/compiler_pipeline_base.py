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
"""Defines the interface for the compiler pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

# Dependency imports

import six


@six.add_metaclass(abc.ABCMeta)
class CompilerPipeline(object):
  """An interface to a pipeline that reduces computations to a simpler form."""

  # TODO(b/113123410): Make this conversion process configurable, and driven
  # largely by what target backend can support. Update the API accordingly.

  @abc.abstractmethod
  def compile(self, computation_proto):
    """Compiles the `computation_proto` into a simpler form for execution.

    Args:
      computation_proto: An instance of the `Computation` proto to compile.

    Returns:
      An instance of the `Computation` proto with logic equivalent to the
       argument `computation_proto`, but in a form that is executable.

    Raises:
      TypeError: If the arguments are of the wrong types.
    """
    raise NotImplementedError
