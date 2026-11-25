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
"""Execution contexts for the native backend."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional
import warnings

import federated_language

from tensorflow_federated.python.core.backends.native import compiler
from tensorflow_federated.python.core.backends.native import mergeable_comp_compiler
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.execution_contexts import mergeable_comp_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import cpp_executor_factory
from tensorflow_federated.python.core.impl.executors import executor_bindings

_STREAM_STRUCTS_LOCAL_DEPRECATION_MESSAGE = (
    'stream_structs is deprecated and has no effect on local in-process '
    'execution contexts. Use a remote C++ execution context to enable struct '
    'streaming.'
)


def create_sync_local_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
    cardinality_inference_fn: Optional[
        Callable[[Any, Any], Mapping[Any, int]]
    ] = None,
) -> federated_language.framework.SyncExecutionContext:
  """Creates a local execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If nonpositive, there is no
      limit.
    stream_structs: Deprecated. Ignored on local in-process contexts.
    cardinality_inference_fn: An optional callable for inferring cardinalities
      of client placements from the arguments to a computation.

  Returns:
    An instance of `federated_language.framework.SyncContext` representing the
    TFF-C++ runtime.
  """
  if stream_structs:
    warnings.warn(
        _STREAM_STRUCTS_LOCAL_DEPRECATION_MESSAGE,
        DeprecationWarning,
        stacklevel=2,
    )
  factory = cpp_executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
  )
  kwargs: dict[str, Any] = {}
  if cardinality_inference_fn is not None:
    kwargs['cardinality_inference_fn'] = cardinality_inference_fn
  context = federated_language.framework.SyncExecutionContext(
      executor_fn=factory,
      compiler_fn=compiler.desugar_and_transform_to_native,
      transform_args=tensorflow_computation.transform_args,
      transform_result=tensorflow_computation.transform_result,
      **kwargs,
  )
  return context


def set_sync_local_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
    cardinality_inference_fn: Optional[
        Callable[[Any, Any], Mapping[Any, int]]
    ] = None,
) -> None:
  """Sets a default local sync execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: Deprecated. Ignored on local in-process contexts.
    cardinality_inference_fn: An optional callable for inferring cardinalities
      of client placements from the arguments to a computation.
  """
  context = create_sync_local_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
      cardinality_inference_fn=cardinality_inference_fn,
  )
  federated_language.framework.set_default_context(context)


def create_async_local_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
    cardinality_inference_fn: Optional[
        Callable[[Any, Any], Mapping[Any, int]]
    ] = None,
) -> federated_language.framework.AsyncExecutionContext:
  """Creates a local async execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: Deprecated. Ignored on local in-process contexts.
    cardinality_inference_fn: An optional callable for inferring cardinalities
      of client placements from the arguments to a computation.

  Returns:
    An instance of `federated_language.framework.AsyncContext` representing the
    TFF-C++ runtime.
  """
  if stream_structs:
    warnings.warn(
        _STREAM_STRUCTS_LOCAL_DEPRECATION_MESSAGE,
        DeprecationWarning,
        stacklevel=2,
    )
  factory = cpp_executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
  )
  kwargs: dict[str, Any] = {}
  if cardinality_inference_fn is not None:
    kwargs['cardinality_inference_fn'] = cardinality_inference_fn
  context = federated_language.framework.AsyncExecutionContext(
      executor_fn=factory,
      compiler_fn=compiler.desugar_and_transform_to_native,
      transform_args=tensorflow_computation.transform_args,
      transform_result=tensorflow_computation.transform_result,
      **kwargs,
  )
  return context


def set_async_local_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
    cardinality_inference_fn: Optional[
        Callable[[Any, Any], Mapping[Any, int]]
    ] = None,
) -> None:
  """Sets a local execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If nonpositive, there is no
      limit.
    stream_structs: Deprecated. Ignored on local in-process contexts.
    cardinality_inference_fn: An optional callable for inferring cardinalities
      of client placements from the arguments to a computation.
  """
  context = create_async_local_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
      cardinality_inference_fn=cardinality_inference_fn,
  )
  federated_language.framework.set_default_context(context)


def create_sync_remote_cpp_execution_context(
    channels: Sequence[executor_bindings.GRPCChannel],
    default_num_clients: int = 0,
    stream_structs: bool = False,
) -> federated_language.framework.SyncExecutionContext:
  """Creates a remote execution context backed by TFF-C++ runtime.

  Args:
    channels: A sequence of gRPC channels connected to remote TFF workers.
    default_num_clients: The number of clients to use as the default cardinality
      if it cannot be inferred from computation arguments.
    stream_structs: If True, decomposes and streams struct values over gRPC.

  Returns:
    An instance of `federated_language.framework.SyncExecutionContext` backed by
    the TFF-C++ runtime.
  """
  factory = cpp_executor_factory.remote_cpp_executor_factory(
      channels=channels,
      default_num_clients=default_num_clients,
      stream_structs=stream_structs,
  )
  context = federated_language.framework.SyncExecutionContext(
      executor_fn=factory,
      compiler_fn=compiler.desugar_and_transform_to_native,
      transform_args=tensorflow_computation.transform_args,
      transform_result=tensorflow_computation.transform_result,
  )
  return context


def set_sync_remote_cpp_execution_context(
    channels: Sequence[executor_bindings.GRPCChannel],
    default_num_clients: int = 0,
    stream_structs: bool = False,
) -> None:
  """Sets a default remote sync execution context backed by TFF-C++ runtime.

  Args:
    channels: A sequence of gRPC channels connected to remote TFF workers.
    default_num_clients: The number of clients to use as the default cardinality
      if it cannot be inferred from computation arguments.
    stream_structs: If True, decomposes and streams struct values over gRPC.
  """
  context = create_sync_remote_cpp_execution_context(
      channels=channels,
      default_num_clients=default_num_clients,
      stream_structs=stream_structs,
  )
  federated_language.framework.set_default_context(context)


def create_async_remote_cpp_execution_context(
    channels: Sequence[executor_bindings.GRPCChannel],
    default_num_clients: int = 0,
    stream_structs: bool = False,
) -> federated_language.framework.AsyncExecutionContext:
  """Creates an async remote execution context backed by TFF-C++ runtime.

  Args:
    channels: A sequence of gRPC channels connected to remote TFF workers.
    default_num_clients: The number of clients to use as the default cardinality
      if it cannot be inferred from computation arguments.
    stream_structs: If True, decomposes and streams struct values over gRPC.

  Returns:
    An instance of `federated_language.framework.AsyncExecutionContext` backed
    by the TFF-C++ runtime.
  """
  factory = cpp_executor_factory.remote_cpp_executor_factory(
      channels=channels,
      default_num_clients=default_num_clients,
      stream_structs=stream_structs,
  )
  context = federated_language.framework.AsyncExecutionContext(
      executor_fn=factory,
      compiler_fn=compiler.desugar_and_transform_to_native,
      transform_args=tensorflow_computation.transform_args,
      transform_result=tensorflow_computation.transform_result,
  )
  return context


def create_mergeable_comp_execution_context(
    async_contexts: Sequence[federated_language.framework.AsyncContext],
    num_subrounds: Optional[int] = None,
) -> mergeable_comp_execution_context.MergeableCompExecutionContext:
  """Creates a context that compiles to and executes mergeable comp form.

  Args:
    async_contexts: Asynchronous TFF execution contexts across which to
      distribute work.
    num_subrounds: An optional integer specifying the total number of subrounds.
      If unspecified, the length of `async_contexts` determines the number of
      subrounds. If more subrounds are requested than contexts are passed,
      invocations will run sequentially. If fewer, the work will run in parallel
      across a subset of `async_contexts`.

  Returns:
    An instance of
    `mergeable_comp_execution_context.MergeableCompExecutionContext`.
  """
  return mergeable_comp_execution_context.MergeableCompExecutionContext(
      async_contexts=async_contexts,
      compiler_fn=mergeable_comp_compiler.compile_to_mergeable_comp_form,
      num_subrounds=num_subrounds,
  )


def set_mergeable_comp_execution_context(
    async_contexts: Sequence[federated_language.framework.AsyncContext],
    num_subrounds: Optional[int] = None,
):
  """Sets a context that compiles to and executes mergeable comp form.

  Args:
    async_contexts: Asynchronous TFF execution contexts across which to
      distribute work.
    num_subrounds: An optional integer specifying the total number of subrounds.
      If unspecified, the length of `async_contexts` determines the number of
      subrounds. If more subrounds are requested than contexts are passed,
      invocations will run sequentially. If fewer, the work will run in parallel
      across a subset of `async_contexts`.
  """
  context = create_mergeable_comp_execution_context(
      async_contexts=async_contexts,
      num_subrounds=num_subrounds,
  )
  federated_language.framework.get_context_stack().set_default_context(context)
