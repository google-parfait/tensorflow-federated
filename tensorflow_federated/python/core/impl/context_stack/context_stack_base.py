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
"""Defines the interface for the context stack."""

import abc
import contextlib


class ContextStack(object, metaclass=abc.ABCMeta):
  """An interface to a context stack for the API to run against."""

  @abc.abstractproperty
  def current(self):
    """Returns the current context (one at the top of the context stack)."""
    raise NotImplementedError

  @contextlib.contextmanager
  @abc.abstractmethod
  def install(self, ctx):
    """A context manager that temporarily installs a new context on the stack.

    The installed context is placed at the top on the stack while in the context
    manager's scope, and remove from the stack upon exiting the scope. This
    method should only be used by the implementation code, and by the unit tests
    for dependency injection.

    Args:
      ctx: The context to temporarily install at the top of the context stack,
        an instance of `Context` defined in `context_base.py`.

    Yields:
      The installed context.

    Raises:
      TypeError: If `ctx` is not a valid instance of `context_base.Context`.
    """
    raise NotImplementedError
