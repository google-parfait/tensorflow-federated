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
"""Defines the interface for the context of execution underlying the API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Context(object):
  """Represents the interface to the context that the API executes against.

  The interfaces in Core API may be used in a variety of contexts, such as at a
  compile time or at runtime, during unit tests, nested inside a body of an
  outer computation being defined or at the top level (not nested), and in the
  former case potentially nested more than one level deep, either in a general
  orchestration scope or in a section of TensorFlow code, as well as potentially
  in an undecorated regular Python function or a tfe.defun. The context of usage
  potentially affects some of the API, determines which subset of the API is
  available, and it can influence the manner in which the API calls behave. This
  interface abstracts out interactions between the API and the underlying
  context for mechanisms that are context-dependent. The API only interacts with
  a given current context (instance). Manipulating a context stack is the
  responsibility of the implementation.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def invoke(self, comp, arg):
    """Invokes computation 'comp' with argument 'arg'.

    Args:
      comp: The computation being invoked. The Python type of 'comp' expected
        here (e.g., pb.Computation. ComputationImpl, or other) may depend on
        the context. It is the responsibility of the concrete implementation
        of this interface to verify that the type of 'comp' matches what the
        context is expecting.
      arg: The optional argument of the call (possibly an argument tuple with
        a nested structure), or None if no argument was supplied. Computations
        accept arguments in a variety of forms, but those are bundled together
        into a single object before invoking the context.

    Returns:
      The result of invocation, which is context-dependent.
    """
    raise NotImplementedError
