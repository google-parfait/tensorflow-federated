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
"""Contexts and constructors for integration testing."""

import asyncio
from collections.abc import Callable
import functools
from typing import Union

from absl.testing import parameterized
import federated_language
import tensorflow_federated as tff


def _create_mergeable_comp_execution_context():
  async_contexts = [
      tff.backends.native.create_async_local_cpp_execution_context()
  ]
  return tff.backends.native.create_mergeable_comp_execution_context(
      async_contexts
  )


def get_all_contexts():
  """Returns a list containing a (name, context_fn) tuple for each context."""
  return [
      ('native_mergeable', _create_mergeable_comp_execution_context),
      (
          'native_sync_local',
          tff.backends.native.create_sync_local_cpp_execution_context,
      ),
      ('test_sync', tff.backends.test.create_sync_test_cpp_execution_context),
  ]


_Context = Union[
    federated_language.framework.AsyncContext,
    federated_language.framework.SyncContext,
]


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
      async def wrapper(self, context_fn: Callable[[], _Context]):
        decorator = federated_language.framework.with_context(context_fn)
        decorated_fn = decorator(fn)
        await decorated_fn(self)

    else:

      @functools.wraps(fn)
      @parameterized.named_parameters(*named_contexts)
      def wrapper(self, context_fn: Callable[[], _Context]):
        decorator = federated_language.framework.with_context(context_fn)
        decorated_fn = decorator(fn)
        decorated_fn(self)

    return wrapper

  return decorator
