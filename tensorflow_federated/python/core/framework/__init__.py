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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.core.impl.computation_building_blocks import Block
from tensorflow_federated.python.core.impl.computation_building_blocks import Call
from tensorflow_federated.python.core.impl.computation_building_blocks import CompiledComputation
from tensorflow_federated.python.core.impl.computation_building_blocks import ComputationBuildingBlock
from tensorflow_federated.python.core.impl.computation_building_blocks import Intrinsic
from tensorflow_federated.python.core.impl.computation_building_blocks import Lambda
from tensorflow_federated.python.core.impl.computation_building_blocks import Placement
from tensorflow_federated.python.core.impl.computation_building_blocks import Reference
from tensorflow_federated.python.core.impl.computation_building_blocks import Selection
from tensorflow_federated.python.core.impl.computation_building_blocks import Tuple
from tensorflow_federated.python.core.impl.computation_constructing_utils import create_federated_map_all_equal
from tensorflow_federated.python.core.impl.computation_constructing_utils import create_federated_map_or_apply
from tensorflow_federated.python.core.impl.computation_constructing_utils import create_federated_zip
from tensorflow_federated.python.core.impl.computation_constructing_utils import unique_name_generator
from tensorflow_federated.python.core.impl.computation_wrapper_instances import building_block_to_computation
from tensorflow_federated.python.core.impl.intrinsic_defs import FEDERATED_AGGREGATE
from tensorflow_federated.python.core.impl.intrinsic_defs import FEDERATED_APPLY
from tensorflow_federated.python.core.impl.intrinsic_defs import FEDERATED_BROADCAST
from tensorflow_federated.python.core.impl.intrinsic_defs import FEDERATED_MAP
from tensorflow_federated.python.core.impl.intrinsic_defs import FEDERATED_MAP_ALL_EQUAL
from tensorflow_federated.python.core.impl.transformation_utils import transform_postorder
from tensorflow_federated.python.core.impl.transformations import check_has_unique_names
from tensorflow_federated.python.core.impl.transformations import check_intrinsics_whitelisted_for_reduction
from tensorflow_federated.python.core.impl.transformations import get_map_of_unbound_references
from tensorflow_federated.python.core.impl.transformations import inline_block_locals
from tensorflow_federated.python.core.impl.transformations import insert_called_tf_identity_at_leaves
from tensorflow_federated.python.core.impl.transformations import is_called_intrinsic
from tensorflow_federated.python.core.impl.transformations import merge_tuple_intrinsics
from tensorflow_federated.python.core.impl.transformations import remove_mapped_or_applied_identity
from tensorflow_federated.python.core.impl.transformations import replace_called_lambda_with_block
from tensorflow_federated.python.core.impl.transformations import replace_selection_from_tuple_with_element
from tensorflow_federated.python.core.impl.transformations import TFParser
from tensorflow_federated.python.core.impl.transformations import uniquify_reference_names
from tensorflow_federated.python.core.impl.transformations import unwrap_placement
from tensorflow_federated.python.core.impl.type_utils import is_assignable_from
from tensorflow_federated.python.core.impl.type_utils import is_tensorflow_compatible_type
from tensorflow_federated.python.core.impl.type_utils import transform_type_postorder
from tensorflow_federated.python.core.impl.type_utils import type_from_tensors
from tensorflow_federated.python.core.impl.type_utils import type_to_tf_tensor_specs
# Used by doc generation script.
_allowed_symbols = [
    "Block",
    "Call",
    "CompiledComputation",
    "ComputationBuildingBlock",
    "FEDERATED_APPLY",
    "FEDERATED_AGGREGATE",
    "FEDERATED_BROADCAST",
    "FEDERATED_MAP",
    "FEDERATED_MAP_ALL_EQUAL",
    "Intrinsic",
    "Lambda",
    "Placement",
    "Reference",
    "Selection",
    "Tuple",
    "check_has_unique_names",
    "check_intrinsics_whitelisted_for_reduction",
    "create_federated_map_all_equal",
    "create_federated_map_or_apply",
    "create_federated_zip",
    "get_map_of_unbound_references",
    "inline_block_locals",
    "insert_called_tf_identity_at_leaves",
    "is_assignable_from",
    "is_called_intrinsic",
    "is_tensorflow_compatible_type",
    "merge_tuple_intrinsics",
    "remove_mapped_or_applied_identity",
    "replace_called_lambda_with_block",
    "replace_selection_from_tuple_with_element",
    "TFParser",
    "building_block_to_computation",
    "transform_postorder",
    "transform_type_postorder",
    "type_from_tensors",
    "type_to_tf_tensor_specs",
    "unique_name_generator",
    "uniquify_reference_names",
    "unwrap_placement",
]
