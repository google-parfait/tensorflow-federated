# Copyright 2019, The TensorFlow Federated Authors.
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
"""Utilities for testing context stacks."""

import asyncio
from collections.abc import Callable, Iterable
import contextlib
import functools
from typing import Optional, Union

from absl.testing import parameterized

from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl

_Context = Union[context_base.AsyncContext, context_base.SyncContext]
_ContextFactory = Callable[[], _Context]
_EnvironmentFactory = Callable[
    [], Iterable[contextlib.AbstractContextManager[None]]
]


class TestContext(context_base.SyncContext):
  """A test context."""

  def invoke(self, comp, arg):
    return NotImplementedError


@contextlib.contextmanager
def test_environment():
  yield None


def with_context(
    context_fn: _ContextFactory,
    environment_fn: Optional[_EnvironmentFactory] = None,
):
  """Returns a decorator for running a test in a context.

  Args:
    context_fn: A `Callable` that constructs a `tff.framework.AsyncContext` or
      `tff.framework.SyncContext` to install beore invoking the decorated
      function.
    environment_fn: A `Callable` that constructs a list of
      `contextlib.AbstractContextManager` to enter before invoking the decorated
      function.
  """

  def decorator(fn):

    @contextlib.contextmanager
    def install_context(
        context_fn: _ContextFactory,
        environment_fn: Optional[_EnvironmentFactory] = None,
    ):
      context = context_fn()
      with context_stack_impl.context_stack.install(context):
        if environment_fn is not None:
          with contextlib.ExitStack() as stack:
            context_managers = environment_fn()
            for context_manager in context_managers:
              stack.enter_context(context_manager)
            yield
        else:
          yield

    if asyncio.iscoroutinefunction(fn):

      @functools.wraps(fn)
      async def wrapper(*args, **kwargs):
        with install_context(context_fn, environment_fn):
          return await fn(*args, **kwargs)

    else:

      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        with install_context(context_fn, environment_fn):
          return fn(*args, **kwargs)

    return wrapper

  return decorator


def with_contexts(*named_contexts):
  """Returns a decorator for parameterizing a test by a context.

  Args:
    *named_contexts: Named parameters used to construct the `with_context`
      decorator; either a single iterable, or a list of `tuple`s or `dict`s.

  Raises:
    ValueError: If no named contexts are passed to the decorator.
  """
  if not named_contexts:
    raise ValueError('Expected at least one named parameter, found none.')

  def decorator(fn):

    if asyncio.iscoroutinefunction(fn):

      @functools.wraps(fn)
      @parameterized.named_parameters(*named_contexts)
      async def wrapper(
          self,
          context_fn: _ContextFactory,
          environment_fn: Optional[_EnvironmentFactory] = None,
      ):
        decorator = with_context(context_fn, environment_fn)
        decorated_fn = decorator(fn)
        await decorated_fn(self)

    else:

      @functools.wraps(fn)
      @parameterized.named_parameters(*named_contexts)
      def wrapper(
          self,
          context_fn: _ContextFactory,
          environment_fn: Optional[_EnvironmentFactory] = None,
      ):
        decorator = with_context(context_fn, environment_fn)
        decorated_fn = decorator(fn)
        decorated_fn(self)

    return wrapper

  return decorator
