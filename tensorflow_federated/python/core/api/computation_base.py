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
"""Defines the abstract interface for classes that represent computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

# Dependency imports

import six


@six.add_metaclass(abc.ABCMeta)
class Computation(object):
  """An abstract interface for all classes that represent computations."""

  @abc.abstractproperty
  def type_signature(self):
    """Provides access to the type signature of this computation.

    Returns:
      An instance of a class that represents this computation's type signature.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def __call__(self, *args, **kwargs):
    """Invokes the computation with the given arguments in the given context.

    Args:
      *args: The positional arguments.
      **kwargs: The keyword-based arguments.

    Returns:
      The result of invoking the computation, the exact form of which depends
      on the context.
    """
    raise NotImplementedError
