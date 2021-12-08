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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Bindings for C++ executor stack construction."""

from typing import Mapping, Sequence

from tensorflow_federated.cc.core.impl.executor_stacks import executor_stack_bindings
from tensorflow_federated.python.core.impl.executors import data_conversions
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.types import placements


def create_remote_executor_stack(
    channels: Sequence[executor_bindings.GRPCChannel],
    cardinalities: Mapping[placements.PlacementLiteral, int]
) -> executor_bindings.Executor:
  """Constructs a RemoteExecutor proxying services on `targets`."""
  uri_cardinalities = data_conversions.convert_cardinalities_dict_to_string_keyed(
      cardinalities)
  return executor_stack_bindings.create_remote_executor_stack(
      channels, uri_cardinalities)
