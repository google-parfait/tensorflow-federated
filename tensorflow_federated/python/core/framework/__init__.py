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

# pylint: disable=g-importing-member
from tensorflow_federated.python.core.impl.compiler.building_block_factory import unique_name_generator
from tensorflow_federated.python.core.impl.compiler.building_blocks import Block
from tensorflow_federated.python.core.impl.compiler.building_blocks import Call
from tensorflow_federated.python.core.impl.compiler.building_blocks import CompiledComputation
from tensorflow_federated.python.core.impl.compiler.building_blocks import ComputationBuildingBlock
from tensorflow_federated.python.core.impl.compiler.building_blocks import Data
from tensorflow_federated.python.core.impl.compiler.building_blocks import Intrinsic
from tensorflow_federated.python.core.impl.compiler.building_blocks import Lambda
from tensorflow_federated.python.core.impl.compiler.building_blocks import Literal
from tensorflow_federated.python.core.impl.compiler.building_blocks import Placement
from tensorflow_federated.python.core.impl.compiler.building_blocks import Reference
from tensorflow_federated.python.core.impl.compiler.building_blocks import Selection
from tensorflow_federated.python.core.impl.compiler.building_blocks import Struct
from tensorflow_federated.python.core.impl.compiler.building_blocks import UnexpectedBlockError
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_AGGREGATE
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_APPLY
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_BROADCAST
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_EVAL_AT_CLIENTS
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_EVAL_AT_SERVER
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_MAP
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_MAP_ALL_EQUAL
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_SUM
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_VALUE_AT_CLIENTS
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_VALUE_AT_SERVER
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_ZIP_AT_CLIENTS
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_ZIP_AT_SERVER
from tensorflow_federated.python.core.impl.compiler.transformation_utils import transform_postorder
from tensorflow_federated.python.core.impl.compiler.transformation_utils import transform_preorder
from tensorflow_federated.python.core.impl.compiler.transformations import to_call_dominant
from tensorflow_federated.python.core.impl.computation.computation_impl import ConcreteComputation
from tensorflow_federated.python.core.impl.computation.computation_serialization import deserialize_computation
from tensorflow_federated.python.core.impl.computation.computation_serialization import serialize_computation
from tensorflow_federated.python.core.impl.context_stack.context_base import AsyncContext
from tensorflow_federated.python.core.impl.context_stack.context_base import SyncContext
from tensorflow_federated.python.core.impl.context_stack.context_stack_base import ContextStack
from tensorflow_federated.python.core.impl.context_stack.get_context_stack import get_context_stack
from tensorflow_federated.python.core.impl.context_stack.set_default_context import set_default_context
from tensorflow_federated.python.core.impl.execution_contexts.async_execution_context import AsyncExecutionContext
from tensorflow_federated.python.core.impl.execution_contexts.mergeable_comp_execution_context import MergeableCompExecutionContext
from tensorflow_federated.python.core.impl.execution_contexts.mergeable_comp_execution_context import MergeableCompForm
from tensorflow_federated.python.core.impl.execution_contexts.sync_execution_context import SyncExecutionContext
from tensorflow_federated.python.core.impl.executor_stacks.executor_factory import local_cpp_executor_factory
from tensorflow_federated.python.core.impl.executor_stacks.python_executor_stacks import ResourceManagingExecutorFactory
from tensorflow_federated.python.core.impl.executors.cardinalities_utils import merge_cardinalities
from tensorflow_federated.python.core.impl.executors.cardinality_carrying_base import CardinalityCarrying
from tensorflow_federated.python.core.impl.executors.data_descriptor import CardinalityFreeDataDescriptor
from tensorflow_federated.python.core.impl.executors.data_descriptor import CreateDataDescriptor
from tensorflow_federated.python.core.impl.executors.data_descriptor import DataDescriptor
from tensorflow_federated.python.core.impl.executors.executor_factory import CardinalitiesType
from tensorflow_federated.python.core.impl.executors.executor_factory import ExecutorFactory
from tensorflow_federated.python.core.impl.executors.executors_errors import RetryableError
from tensorflow_federated.python.core.impl.executors.ingestable_base import Ingestable
from tensorflow_federated.python.core.impl.executors.remote_executor import RemoteExecutor
from tensorflow_federated.python.core.impl.executors.remote_executor_grpc_stub import RemoteExecutorGrpcStub
from tensorflow_federated.python.core.impl.executors.remote_executor_stub import RemoteExecutorStub
from tensorflow_federated.python.core.impl.executors.value_serialization import deserialize_value
from tensorflow_federated.python.core.impl.executors.value_serialization import serialize_value
from tensorflow_federated.python.core.impl.types.placements import PlacementLiteral
# pylint: enable=g-importing-member
