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
"""Bindings for C++ executor stack construction."""

from collections.abc import Mapping, Sequence

import federated_language
import federated_language_executor

from tensorflow_federated.cc.core.impl.executor_stacks import executor_stack_bindings


def filter_to_live_channels(
    channels: Sequence[federated_language_executor.GRPCChannel],
    wait_connected_duration_millis: int = 1000,
) -> Sequence[federated_language_executor.GRPCChannel]:
  """Waits and filters channels that are ready or idle."""
  return executor_stack_bindings.filter_to_live_channels(
      channels, wait_connected_duration_millis
  )


def create_remote_executor_stack(
    channels: Sequence[federated_language_executor.GRPCChannel],
    cardinalities: Mapping[federated_language.framework.PlacementLiteral, int],
    max_concurrent_computation_calls: int = -1,
) -> federated_language_executor.Executor:
  """Constructs a RemoteExecutor proxying services on `targets`."""
  uri_cardinalities = (
      federated_language_executor.convert_cardinalities_dict_to_string_keyed(
          cardinalities
      )
  )
  return executor_stack_bindings.create_remote_executor_stack(
      channels, uri_cardinalities, max_concurrent_computation_calls
  )


def create_streaming_remote_executor_stack(
    channels: Sequence[federated_language_executor.GRPCChannel],
    cardinalities: Mapping[federated_language.framework.PlacementLiteral, int],
) -> federated_language_executor.Executor:
  """Constructs a RemoteExecutor proxying services on `targets`."""
  uri_cardinalities = (
      federated_language_executor.convert_cardinalities_dict_to_string_keyed(
          cardinalities
      )
  )
  return executor_stack_bindings.create_streaming_remote_executor_stack(
      channels, uri_cardinalities
  )
