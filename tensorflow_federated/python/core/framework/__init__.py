# Lint as: python3
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
"""Interfaces for extensions, selectively lifted out of `impl`."""

from tensorflow_federated.python.core.impl.compiler.building_block_analysis import is_called_intrinsic
from tensorflow_federated.python.core.impl.compiler.building_block_factory import create_federated_map_all_equal
from tensorflow_federated.python.core.impl.compiler.building_block_factory import create_federated_map_or_apply
from tensorflow_federated.python.core.impl.compiler.building_block_factory import create_federated_zip
from tensorflow_federated.python.core.impl.compiler.building_block_factory import unique_name_generator
from tensorflow_federated.python.core.impl.compiler.building_blocks import Block
from tensorflow_federated.python.core.impl.compiler.building_blocks import Call
from tensorflow_federated.python.core.impl.compiler.building_blocks import CompiledComputation
from tensorflow_federated.python.core.impl.compiler.building_blocks import ComputationBuildingBlock
from tensorflow_federated.python.core.impl.compiler.building_blocks import Intrinsic
from tensorflow_federated.python.core.impl.compiler.building_blocks import Lambda
from tensorflow_federated.python.core.impl.compiler.building_blocks import Placement
from tensorflow_federated.python.core.impl.compiler.building_blocks import Reference
from tensorflow_federated.python.core.impl.compiler.building_blocks import Selection
from tensorflow_federated.python.core.impl.compiler.building_blocks import Tuple
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_AGGREGATE
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_APPLY
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_BROADCAST
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_MAP
from tensorflow_federated.python.core.impl.compiler.intrinsic_defs import FEDERATED_MAP_ALL_EQUAL
from tensorflow_federated.python.core.impl.compiler.transformation_utils import get_map_of_unbound_references
from tensorflow_federated.python.core.impl.compiler.transformation_utils import transform_postorder
from tensorflow_federated.python.core.impl.compiler.transformations import remove_lambdas_and_blocks
from tensorflow_federated.python.core.impl.compiler.tree_analysis import check_broadcast_not_dependent_on_aggregate
from tensorflow_federated.python.core.impl.compiler.tree_analysis import check_has_unique_names
from tensorflow_federated.python.core.impl.compiler.tree_analysis import check_intrinsics_whitelisted_for_reduction
from tensorflow_federated.python.core.impl.compiler.tree_transformations import inline_block_locals
from tensorflow_federated.python.core.impl.compiler.tree_transformations import insert_called_tf_identity_at_leaves
from tensorflow_federated.python.core.impl.compiler.tree_transformations import merge_tuple_intrinsics
from tensorflow_federated.python.core.impl.compiler.tree_transformations import remove_mapped_or_applied_identity
from tensorflow_federated.python.core.impl.compiler.tree_transformations import replace_called_lambda_with_block
from tensorflow_federated.python.core.impl.compiler.tree_transformations import uniquify_reference_names
from tensorflow_federated.python.core.impl.compiler.tree_transformations import unwrap_placement
from tensorflow_federated.python.core.impl.compiler.type_serialization import deserialize_type
from tensorflow_federated.python.core.impl.compiler.type_serialization import serialize_type
from tensorflow_federated.python.core.impl.context_stack.context_base import Context
from tensorflow_federated.python.core.impl.context_stack.context_stack_base import ContextStack
from tensorflow_federated.python.core.impl.context_stack.get_context_stack import get_context_stack
from tensorflow_federated.python.core.impl.context_stack.set_default_context import set_default_context
from tensorflow_federated.python.core.impl.executors.caching_executor import CachingExecutor
from tensorflow_federated.python.core.impl.executors.composing_executor import ComposingExecutor
from tensorflow_federated.python.core.impl.executors.default_executor import set_default_executor
from tensorflow_federated.python.core.impl.executors.eager_tf_executor import EagerTFExecutor
from tensorflow_federated.python.core.impl.executors.executor_base import Executor
from tensorflow_federated.python.core.impl.executors.executor_factory import create_executor_factory
from tensorflow_federated.python.core.impl.executors.executor_factory import ExecutorFactory
from tensorflow_federated.python.core.impl.executors.executor_service import ExecutorService
from tensorflow_federated.python.core.impl.executors.executor_stacks import local_executor_factory
from tensorflow_federated.python.core.impl.executors.executor_stacks import sizing_executor_factory
from tensorflow_federated.python.core.impl.executors.executor_stacks import worker_pool_executor_factory
from tensorflow_federated.python.core.impl.executors.executor_value_base import ExecutorValue
from tensorflow_federated.python.core.impl.executors.federating_executor import FederatingExecutor
from tensorflow_federated.python.core.impl.executors.reference_resolving_executor import ReferenceResolvingExecutor
from tensorflow_federated.python.core.impl.executors.remote_executor import RemoteExecutor
from tensorflow_federated.python.core.impl.executors.thread_delegating_executor import ThreadDelegatingExecutor
from tensorflow_federated.python.core.impl.executors.transforming_executor import TransformingExecutor
from tensorflow_federated.python.core.impl.reference_executor import ReferenceExecutor
from tensorflow_federated.python.core.impl.tree_to_cc_transformations import TFParser
from tensorflow_federated.python.core.impl.type_utils import are_equivalent_types
from tensorflow_federated.python.core.impl.type_utils import is_assignable_from
from tensorflow_federated.python.core.impl.type_utils import is_tensorflow_compatible_type
from tensorflow_federated.python.core.impl.type_utils import transform_type_postorder
from tensorflow_federated.python.core.impl.type_utils import type_from_tensors
from tensorflow_federated.python.core.impl.type_utils import type_to_tf_tensor_specs
from tensorflow_federated.python.core.impl.wrappers.computation_wrapper_instances import building_block_to_computation

# Used by doc generation script.
_allowed_symbols = [
    "Block",
    "CachingExecutor",
    "Call",
    "CompiledComputation",
    "ComposingExecutor",
    "ComputationBuildingBlock",
    "ThreadDelegatingExecutor",
    "Context",
    "ContextStack",
    "EagerTFExecutor",
    "Executor",
    "ExecutorService",
    "ExecutorValue",
    "FEDERATED_AGGREGATE",
    "FEDERATED_APPLY",
    "FEDERATED_BROADCAST",
    "FEDERATED_MAP",
    "FEDERATED_MAP_ALL_EQUAL",
    "FederatingExecutor",
    "Intrinsic",
    "Lambda",
    "ReferenceResolvingExecutor",
    "Placement",
    "Reference",
    "RemoteExecutor",
    "Selection",
    "TFParser",
    "TransformingExecutor",
    "Tuple",
    "are_equivalent_types",
    "building_block_to_computation",
    "check_has_unique_names",
    "check_intrinsics_whitelisted_for_reduction",
    "create_executor_factory",
    "create_federated_map_all_equal",
    "create_federated_map_or_apply",
    "create_federated_zip",
    "deserialize_type",
    "get_context_stack",
    "get_map_of_unbound_references",
    "inline_block_locals",
    "insert_called_tf_identity_at_leaves",
    "is_assignable_from",
    "is_called_intrinsic",
    "is_tensorflow_compatible_type",
    "local_executor_factory",
    "merge_tuple_intrinsics",
    "remove_lambdas_and_blocks",
    "remove_mapped_or_applied_identity",
    "replace_called_lambda_with_block",
    "serialize_type",
    "set_default_context",
    "set_default_executor",
    "transform_postorder",
    "transform_type_postorder",
    "type_from_tensors",
    "type_to_tf_tensor_specs",
    "unique_name_generator",
    "uniquify_reference_names",
    "unwrap_placement",
    "worker_pool_executor_factory",
]
