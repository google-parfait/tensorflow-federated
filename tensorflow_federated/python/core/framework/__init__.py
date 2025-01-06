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
"""Libraries for extending the TensorFlow Federated core library."""

import federated_language
# pylint: disable=g-importing-member
unique_name_generator = federated_language.framework.unique_name_generator
Block = federated_language.framework.Block
Call = federated_language.framework.Call
CompiledComputation = federated_language.framework.CompiledComputation
ComputationBuildingBlock = federated_language.framework.ComputationBuildingBlock
Data = federated_language.framework.Data
Intrinsic = federated_language.framework.Intrinsic
Lambda = federated_language.framework.Lambda
Literal = federated_language.framework.Literal
Placement = federated_language.framework.Placement
Reference = federated_language.framework.Reference
Selection = federated_language.framework.Selection
Struct = federated_language.framework.Struct
UnexpectedBlockError = federated_language.framework.UnexpectedBlockError
FEDERATED_AGGREGATE = federated_language.framework.FEDERATED_AGGREGATE
FEDERATED_APPLY = federated_language.framework.FEDERATED_APPLY
FEDERATED_BROADCAST = federated_language.framework.FEDERATED_BROADCAST
FEDERATED_EVAL_AT_CLIENTS = (
    federated_language.framework.FEDERATED_EVAL_AT_CLIENTS
)
FEDERATED_EVAL_AT_SERVER = federated_language.framework.FEDERATED_EVAL_AT_SERVER
FEDERATED_MAP = federated_language.framework.FEDERATED_MAP
FEDERATED_MAP_ALL_EQUAL = federated_language.framework.FEDERATED_MAP_ALL_EQUAL
FEDERATED_SUM = federated_language.framework.FEDERATED_SUM
FEDERATED_VALUE_AT_CLIENTS = (
    federated_language.framework.FEDERATED_VALUE_AT_CLIENTS
)
FEDERATED_VALUE_AT_SERVER = (
    federated_language.framework.FEDERATED_VALUE_AT_SERVER
)
FEDERATED_ZIP_AT_CLIENTS = federated_language.framework.FEDERATED_ZIP_AT_CLIENTS
FEDERATED_ZIP_AT_SERVER = federated_language.framework.FEDERATED_ZIP_AT_SERVER
transform_postorder = federated_language.framework.transform_postorder
transform_preorder = federated_language.framework.transform_preorder
from tensorflow_federated.python.core.impl.compiler.transformations import to_call_dominant

ConcreteComputation = federated_language.framework.ConcreteComputation
deserialize_computation = federated_language.framework.deserialize_computation
serialize_computation = federated_language.framework.serialize_computation
pack_args_into_struct = federated_language.framework.pack_args_into_struct
unpack_args_from_struct = federated_language.framework.unpack_args_from_struct
AsyncContext = federated_language.framework.AsyncContext
SyncContext = federated_language.framework.SyncContext
ContextStack = federated_language.framework.ContextStack
get_context_stack = federated_language.framework.get_context_stack
set_default_context = federated_language.framework.set_default_context
from tensorflow_federated.python.core.impl.execution_contexts.mergeable_comp_execution_context import MergeableCompExecutionContext
from tensorflow_federated.python.core.impl.execution_contexts.mergeable_comp_execution_context import MergeableCompForm
from tensorflow_federated.python.core.impl.executor_stacks.executor_factory import local_cpp_executor_factory
from tensorflow_federated.python.core.impl.executor_stacks.python_executor_stacks import ResourceManagingExecutorFactory
from tensorflow_federated.python.core.impl.executors.remote_executor import RemoteExecutor
from tensorflow_federated.python.core.impl.executors.remote_executor_grpc_stub import RemoteExecutorGrpcStub
from tensorflow_federated.python.core.impl.executors.remote_executor_stub import RemoteExecutorStub
from tensorflow_federated.python.core.impl.executors.value_serialization import deserialize_value
from tensorflow_federated.python.core.impl.executors.value_serialization import serialize_value

PlacementLiteral = federated_language.framework.PlacementLiteral
# pylint: enable=g-importing-member
