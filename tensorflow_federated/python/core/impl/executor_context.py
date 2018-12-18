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
"""A context in which computation calls can be handled by various executors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import compiler_pipeline_base
from tensorflow_federated.python.core.impl import context_base
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import federated_computation_context
from tensorflow_federated.python.core.impl import value_impl


class ExecutorContext(context_base.Context):
  """An implementation of context that routes computation calls to executors."""

  # TODO(b/113116813): Plug this in as a default with the reference executor at
  # the top of the context stack, so it does not have to be explicitly created
  # in tests or colab notebooks.

  def __init__(self, executor, context_stack, compiler_pipeline):
    """Constructs this context with a given target `executor`.

    Args:
      executor: An instance of `executor_base.Executor` to handle calls.
      context_stack: The context stack to use.
      compiler_pipeline: The compiler pipeline to use.
    """
    py_typecheck.check_type(executor, executor_base.Executor)
    py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
    py_typecheck.check_type(compiler_pipeline,
                            compiler_pipeline_base.CompilerPipeline)
    self._executor = executor
    self._context_stack = context_stack
    self._compiler_pipeline = compiler_pipeline

  def invoke(self, comp, arg):
    # Bake arguments into the call if needed and produce a self-contained proto
    # before submitting for execution.
    context = federated_computation_context.FederatedComputationContext(
        self._context_stack)
    result = context.invoke(comp, arg)
    computation_proto = value_impl.ValueImpl.get_comp(result).proto
    return self._executor.execute(
        self._compiler_pipeline.compile(computation_proto))
