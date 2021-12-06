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

import contextlib
import functools

from absl.testing import parameterized
import portpicker
import tensorflow_federated as tff

from tensorflow_federated.python.tests import remote_runtime_test_utils

WORKER_PORTS = [portpicker.pick_unused_port() for _ in range(2)]
AGGREGATOR_PORTS = [portpicker.pick_unused_port() for _ in range(2)]


def _create_local_mergeable_comp_context():
  factory = tff.framework.local_executor_factory()
  return tff.backends.native.create_mergeable_comp_execution_context([factory])


def create_sequence_op_supporting_context():
  executor_factory = tff.framework.local_executor_factory(
      support_sequence_ops=True)
  return tff.framework.ExecutionContext(
      executor_fn=executor_factory,
      compiler_fn=tff.backends.native.compiler.transform_to_native_form)  # pytype: disable=wrong-arg-types


def _get_all_contexts():
  """Returns a list containing a (name, context_fn) tuple for each context."""
  # pyformat: disable
  return [
      # pylint: disable=unnecessary-lambda
      # native_local_cpp removed by copybara
      ('native_local', lambda: tff.backends.native.create_local_python_execution_context()),
      ('native_mergeable', lambda: _create_local_mergeable_comp_context()),
      ('native_remote',
       lambda: remote_runtime_test_utils.create_localhost_remote_context(WORKER_PORTS),
       lambda: remote_runtime_test_utils.create_inprocess_worker_contexts(WORKER_PORTS)),
      ('native_remote_intermediate_aggregator',
       lambda: remote_runtime_test_utils.create_localhost_remote_context(AGGREGATOR_PORTS),
       lambda: remote_runtime_test_utils.create_inprocess_aggregator_contexts(WORKER_PORTS, AGGREGATOR_PORTS)),
      ('native_sizing', lambda: tff.backends.native.create_sizing_execution_context()),
      ('native_thread_debug',
       lambda: tff.backends.native.create_thread_debugging_execution_context()),
      ('test', lambda: tff.backends.test.create_test_execution_context()),
  ]
  # pyformat: enable


def with_context(context_fn):
  """A decorator for running tests in the given `context`."""

  def decorator_context(fn):

    @functools.wraps(fn)
    def wrapper_context(self):
      context_stack = tff.framework.get_context_stack()
      with context_stack.install(context_fn()):
        fn(self)

    return wrapper_context

  return decorator_context


def with_environment(server_contexts_fn):
  """A decorator for running tests in an environment."""

  def decorator_environment(fn):

    @functools.wraps(fn)
    def wrapper_environment(self):
      with contextlib.ExitStack() as stack:
        for server_context in server_contexts_fn():
          stack.enter_context(server_context)
        fn(self)

    return wrapper_environment

  return decorator_environment


def with_contexts(*args):
  """A decorator for creating tests parameterized by context."""

  def decorator_contexts(fn, *named_contexts):
    if not named_contexts:
      named_contexts = _get_all_contexts()

    @parameterized.named_parameters(*named_contexts)
    def wrapper_contexts(self, context_fn, server_contexts_fn=None):
      with_context_decorator = with_context(context_fn)
      decorated_fn = with_context_decorator(fn)
      if server_contexts_fn is not None:
        with_environment_decorator = with_environment(server_contexts_fn)
        decorated_fn = with_environment_decorator(decorated_fn)
      decorated_fn(self)

    return wrapper_contexts

  if len(args) == 1 and callable(args[0]):
    return decorator_contexts(args[0])
  else:
    return lambda fn: decorator_contexts(fn, *args)
