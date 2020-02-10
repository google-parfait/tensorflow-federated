# Lint as: python3
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
"""General purpose test utilities for TFF."""

from absl import logging
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import context_base
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import execution_context
from tensorflow_federated.python.core.impl import executor_stacks
from tensorflow_federated.python.core.impl import reference_executor

tf.compat.v1.enable_v2_behavior()  # Required to create a local executor.


def tf1_and_tf2(fn):
  """A decorator for creating test parameterized by TF computation decorators.

  Args:
    fn: A test function to be decorated. It must accept two arguments: self (a
      `TestCase`), and tf_computation (either a `tff.tf_computation` or
      `tff.tf2_computation`).

  Returns:
    A decorated function, which executes `fn` using both decorators.
  """

  def wrapped_fn(self):
    logging.info('Testing under tff.tf2_computation')
    fn(self, computations.tf2_computation)
    logging.info('Testing under tff.tf_computation')
    fn(self, computations.tf_computation)

  return wrapped_fn


def tf1(fn):
  """A decorator for testing the `tff.tf_computation` decorator."""

  def wrapped_fn(self):
    fn(self, computations.tf_computation)

  return wrapped_fn


def tf2(fn):
  """A decorator for testing the `tff.tf2_computation` decorator."""

  def wrapped_fn(self):
    fn(self, computations.tf2_computation)

  return wrapped_fn


def executors(*args):
  """A decorator for creating tests parameterized by executors.

  NOTE: To use this decorator your test is required to inherit from
  `parameterized.TestCase`.

  1.  The decorator can be specified without arguments:

      ```
      @executors
      def foo(self):
        ...
      ```

  2.  The decorator can be called with arguments:

      ```
      @executors(
        ('label', executor),
        ...
      )
      def foo(self):
        ...
      ```

  If the decorator is specified without arguments or is called with no
  arguments, the default this decorator with parameterize the test by the
  following executors:

  *   reference executor
  *   local executor

  If the decorator is called with arguments the arguments must be in a form that
  is accpeted by `parameterized.named_parameters`.

  Args:
    *args: Either a test function to be decorated or named executors for the
      decorated method, either a single iterable, or a list of tuples or dicts.

  Returns:
     A test generator to be handled by `parameterized.TestGeneratorMetaclass`.
  """

  def executor_decorator(fn):
    """Create a wrapped function with custom execution contexts."""

    def wrapped_fn(self, executor):
      """Install a particular execution context before running `fn`."""
      # Executors inheriting from `executor_base.Executor` will need to be
      # wrapped in an execution context. The `ReferenceExecutor` is special and
      # inherits from `context_base.Context`, so we don't wrap.
      if not isinstance(executor, context_base.Context):
        context = execution_context.ExecutionContext(executor)
      else:
        context = executor
      with context_stack_impl.context_stack.install(context):
        fn(self)

    return wrapped_fn

  def decorator(fn, named_executors=None):
    """Construct a custom `parameterized.named_parameter` decorator for `fn`."""
    if not named_executors:
      named_executors = [
          ('reference', reference_executor.ReferenceExecutor(compiler=None)),
          ('local', executor_stacks.local_executor_factory()),
      ]
    named_parameters_decorator = parameterized.named_parameters(named_executors)
    fn = executor_decorator(fn)
    fn = named_parameters_decorator(fn)
    return fn

  if len(args) == 1 and callable(args[0]):
    return decorator(args[0])
  else:
    return lambda x: decorator(x, *args)
