# Copyright 2022, The TensorFlow Federated Authors.
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
"""A context for execution based on an embedded executor instance."""

import asyncio

from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.execution_contexts import cpp_async_execution_context


class SyncSerializeAndExecuteCPPContext(context_base.Context):
  """A synchronous execution context delegating to CPP Executor bindings."""

  def __init__(self, factory, compiler_fn):
    self._async_execution_context = cpp_async_execution_context.AsyncSerializeAndExecuteCPPContext(
        factory, compiler_fn)
    self._loop = asyncio.new_event_loop()

  def ingest(self, val, type_spec):
    return self._loop.run_until_complete(
        self._async_execution_context.ingest(val, type_spec))

  def invoke(self, comp, arg):
    return self._loop.run_until_complete(
        self._async_execution_context.invoke(comp, arg))
