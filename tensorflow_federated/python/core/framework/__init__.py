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
from tensorflow_federated.python.core.impl.computation_constructing_utils import unique_name_generator
from tensorflow_federated.python.core.impl.intrinsic_defs import FEDERATED_AGGREGATE
from tensorflow_federated.python.core.impl.intrinsic_defs import FEDERATED_BROADCAST
from tensorflow_federated.python.core.impl.transformation_utils import transform_postorder
from tensorflow_federated.python.core.impl.transformations import _check_has_unique_names as check_has_unique_names
from tensorflow_federated.python.core.impl.transformations import _get_map_of_unbound_references as get_map_of_unbound_references
from tensorflow_federated.python.core.impl.transformations import _is_called_intrinsic as is_called_intrinsic
from tensorflow_federated.python.core.impl.transformations import inline_block_locals
from tensorflow_federated.python.core.impl.transformations import merge_tuple_intrinsics
from tensorflow_federated.python.core.impl.transformations import replace_called_lambda_with_block
from tensorflow_federated.python.core.impl.transformations import uniquify_reference_names
from tensorflow_federated.python.core.impl.type_utils import is_assignable_from
from tensorflow_federated.python.core.impl.type_utils import type_to_tf_tensor_specs

# Used by doc generation script.
_allowed_symbols = [
    "Block",
    "Call",
    "CompiledComputation",
    "ComputationBuildingBlock",
    "FEDERATED_AGGREGATE",
    "FEDERATED_BROADCAST",
    "Intrinsic",
    "Lambda",
    "Placement",
    "Reference",
    "Selection",
    "Tuple",
    "check_has_unique_names",
    "get_map_of_unbound_references",
    "inline_block_locals",
    "is_assignable_from",
    "is_called_intrinsic",
    "merge_tuple_intrinsics",
    "replace_called_lambda_with_block",
    "transform_postorder",
    "type_to_tf_tensor_specs",
    "unique_name_generator",
    "uniquify_reference_names",
]
