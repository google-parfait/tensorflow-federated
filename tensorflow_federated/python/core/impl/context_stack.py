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
"""Defines classes/functions to manipulate the API context stack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager

import threading

from tensorflow_federated.python.core.impl import context_base
from tensorflow_federated.python.core.impl import default_context


class ContextStack(threading.local):
  """A thread-local stack of contexts for the API to execute against."""

  def __init__(self):
    super(ContextStack, self).__init__()
    self._stack = [default_context.DefaultContext()]

  @property
  def current(self):
    """Returns the current context (one at the top of the context stack)."""
    assert self._stack
    ctx = self._stack[-1]
    assert isinstance(ctx, context_base.Context)
    return ctx

  @contextmanager
  def install(self, ctx):
    """A context manager that temporarily installs a new context on the stack.

    The installed context is placed at the top on the stack while in the context
    manager's scope, and remove from the stack upon exiting the scope. This
    method should only be used by the implementation code, and by the unit tests
    for dependency injection.

    Args:
      ctx: The context to temporarily install at the top of the context stack.

    Yields:
      The installed context.

    Raises:
      TypeError: if 'ctx' is not a valid inastance of context_base.Context.
    """
    if not isinstance(ctx, context_base.Context):
      raise TypeError('Expected the context to be of type {}, found {}.'.format(
          context_base.Context.__name__, type(ctx).__name__))
    self._stack.append(ctx)
    try:
      yield ctx
    finally:
      self._stack.pop()


context_stack = ContextStack()
