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
"""Utils for testing executors."""


from absl.testing import parameterized

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import reference_executor
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_value_base


def install_executor(executor_factory_instance):
  context = execution_context.ExecutionContext(executor_factory_instance)
  return context_stack_impl.context_stack.install(context)


def executors(*args):
  """A decorator for creating tests parameterized by executors.

  Note: To use this decorator your test is required to inherit from
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

  def decorator(fn, *named_executors):
    """Construct a custom `parameterized.named_parameter` decorator for `fn`."""
    if not named_executors:
      named_executors = [
          ('reference', reference_executor.ReferenceExecutor(compiler=None)),
          ('local', executor_stacks.local_executor_factory()),
      ]
    named_parameters_decorator = parameterized.named_parameters(
        *named_executors)
    fn = executor_decorator(fn)
    fn = named_parameters_decorator(fn)
    return fn

  if len(args) == 1 and callable(args[0]):
    return decorator(args[0])
  else:
    return lambda x: decorator(x, *args)


class TracingExecutor(executor_base.Executor):
  """Tracing executor keeps a log of all calls for use in testing."""

  def __init__(self, target):
    """Creates a new instance of a tracing executor.

    The tracing executor keeps the trace of all calls. Entries in the trace
    consist of the method name followed by arguments and the returned result,
    with the executor values represented as integer indexes starting from 1.

    Args:
      target: An instance of `executor_base.Executor`.
    """
    py_typecheck.check_type(target, executor_base.Executor)
    self._target = target
    self._last_used_index = 0
    self._trace = []

  @property
  def trace(self):
    return self._trace

  def _get_new_value_index(self):
    val_index = self._last_used_index + 1
    self._last_used_index = val_index
    return val_index

  async def create_value(self, value, type_spec=None):
    target_val = await self._target.create_value(value, type_spec)
    wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                       target_val)
    if type_spec is not None:
      self._trace.append(('create_value', value, type_spec, wrapped_val.index))
    else:
      self._trace.append(('create_value', value, wrapped_val.index))
    return wrapped_val

  async def create_call(self, comp, arg=None):
    if arg is not None:
      target_val = await self._target.create_call(comp.value, arg.value)
      wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                         target_val)
      self._trace.append(
          ('create_call', comp.index, arg.index, wrapped_val.index))
      return wrapped_val
    else:
      target_val = await self._target.create_call(comp.value)
      wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                         target_val)
      self._trace.append(('create_call', comp.index, wrapped_val.index))
      return wrapped_val

  async def create_tuple(self, elements):
    target_val = await self._target.create_tuple(
        anonymous_tuple.map_structure(lambda x: x.value, elements))
    wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                       target_val)
    self._trace.append(
        ('create_tuple',
         anonymous_tuple.map_structure(lambda x: x.index,
                                       elements), wrapped_val.index))
    return wrapped_val

  def close(self):
    self._target.close()

  async def create_selection(self, source, index=None, name=None):
    target_val = await self._target.create_selection(
        source.value, index=index, name=name)
    wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                       target_val)
    self._trace.append(
        ('create_selection', source.index, index if index is not None else name,
         wrapped_val.index))
    return wrapped_val


class TracingExecutorValue(executor_value_base.ExecutorValue):
  """A value managed by `TracingExecutor`."""

  def __init__(self, owner, index, value):
    """Creates an instance of a value in the tracing executor.

    Args:
      owner: An instance of `TracingExecutor`.
      index: An integer identifying the value.
      value: An embedded value from the target executor.
    """
    py_typecheck.check_type(owner, TracingExecutor)
    py_typecheck.check_type(index, int)
    py_typecheck.check_type(value, executor_value_base.ExecutorValue)
    self._owner = owner
    self._index = index
    self._value = value

  @property
  def index(self):
    return self._index

  @property
  def value(self):
    return self._value

  @property
  def type_signature(self):
    return self._value.type_signature

  async def compute(self):
    result = await self._value.compute()
    self._owner.trace.append(('compute', self._index, result))
    return result
