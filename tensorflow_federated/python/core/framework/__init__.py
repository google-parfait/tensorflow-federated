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
from tensorflow_federated.python.core.impl.type_utils import is_assignable_from
from tensorflow_federated.python.core.impl.type_utils import type_to_tf_tensor_specs

# Used by doc generation script.
_allowed_symbols = [
    "Block",
    "Call",
    "CompiledComputation",
    "ComputationBuildingBlock",
    "Intrinsic",
    "Lambda",
    "Placement",
    "Reference",
    "Selection",
    "Tuple",
    "is_assignable_from",
    "type_to_tf_tensor_specs",
]
