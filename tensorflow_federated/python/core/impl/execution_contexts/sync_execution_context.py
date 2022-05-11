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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""A context for execution based on an embedded executor instance."""

import asyncio
from typing import Any
from typing import Callable
from typing import Optional

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.executors import cardinalities_utils
from tensorflow_federated.python.core.impl.executors import executor_factory


class ExecutionContext(context_base.Context):
  """A synchronous execution context backed by an `executor_base.Executor`."""

  def __init__(
      self,
      executor_fn: executor_factory.ExecutorFactory,
      compiler_fn: Optional[Callable[[computation_base.Computation],
                                     Any]] = None,
      *,
      cardinality_inference_fn: cardinalities_utils
      .CardinalityInferenceFnType = cardinalities_utils.infer_cardinalities):
    """Initializes a synchronous execution context which retries invocations.

    Args:
      executor_fn: Instance of `executor_factory.ExecutorFactory`.
      compiler_fn: A Python function that will be used to compile a computation.
      cardinality_inference_fn: A Python function specifying how to infer
        cardinalities from arguments (and their associated types). The value
        returned by this function will be passed to the `create_executor` method
        of `executor_fn` to construct a `tff.framework.Executor` instance.
    """
    py_typecheck.check_type(executor_fn, executor_factory.ExecutorFactory)
    self._executor_factory = executor_fn
    self._async_context = async_execution_context.AsyncExecutionContext(
        executor_fn=executor_fn,
        compiler_fn=compiler_fn,
        cardinality_inference_fn=cardinality_inference_fn)

    self._event_loop = asyncio.new_event_loop()
    self._event_loop.set_task_factory(
        tracing.propagate_trace_context_task_factory)

  @property
  def executor_factory(self):
    return self._executor_factory

  def invoke(self, comp, arg):
    return self._event_loop.run_until_complete(
        self._async_context.invoke(comp, arg))
