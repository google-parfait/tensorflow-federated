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

import contextlib
import threading

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_base
from tensorflow_federated.python.core.impl.context_stack import runtime_error_context


class ContextStackImpl(context_stack_base.ContextStack, threading.local):
  """An implementation of a common thread-local context stack to run against."""

  def __init__(self, default_context):
    super().__init__()
    self._stack = [default_context]

  def set_default_context(self, ctx):
    """Places `ctx` at the bottom of the stack.

    Args:
      ctx: An instance of `context_base.Context`.
    """
    py_typecheck.check_type(ctx, context_base.Context)
    assert self._stack
    self._stack[0] = ctx

  @property
  def current(self):
    assert self._stack
    ctx = self._stack[-1]
    assert isinstance(ctx, context_base.Context)
    return ctx

  @contextlib.contextmanager
  def install(self, ctx):
    py_typecheck.check_type(ctx, context_base.Context)
    self._stack.append(ctx)
    try:
      yield ctx
    finally:
      self._stack.pop()


context_stack = ContextStackImpl(runtime_error_context.RuntimeErrorContext())
