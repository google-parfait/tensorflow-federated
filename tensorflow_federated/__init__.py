# Copyright 2019, The TensorFlow Federated Authors.
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
"""The TensorFlow Federated library."""

import sys

from tensorflow_federated.version import __version__  # pylint: disable=g-bad-import-order

from tensorflow_federated import experimental
from tensorflow_federated.python import aggregators
from tensorflow_federated.python import analytics
from tensorflow_federated.python import learning
from tensorflow_federated.python import program
from tensorflow_federated.python import simulation
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import tracing as profiler
from tensorflow_federated.python.core import backends
from tensorflow_federated.python.core import framework
from tensorflow_federated.python.core import templates
from tensorflow_federated.python.core import test
from tensorflow_federated.python.core import types
from tensorflow_federated.python.core.api.computation_base import Computation
from tensorflow_federated.python.core.api.computations import check_returns_type
from tensorflow_federated.python.core.api.computations import federated_computation
from tensorflow_federated.python.core.api.computations import tf_computation
from tensorflow_federated.python.core.impl.federated_context.data import data
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_aggregate
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_broadcast
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_collect
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_eval
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_map
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_mean
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_secure_modular_sum
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_secure_select
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_secure_sum
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_secure_sum_bitwidth
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_select
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_sum
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_value
from tensorflow_federated.python.core.impl.federated_context.intrinsics import federated_zip
from tensorflow_federated.python.core.impl.federated_context.intrinsics import sequence_map
from tensorflow_federated.python.core.impl.federated_context.intrinsics import sequence_reduce
from tensorflow_federated.python.core.impl.federated_context.intrinsics import sequence_sum
from tensorflow_federated.python.core.impl.federated_context.value_impl import to_value
from tensorflow_federated.python.core.impl.federated_context.value_impl import Value
from tensorflow_federated.python.core.impl.types.computation_types import at_clients as type_at_clients
from tensorflow_federated.python.core.impl.types.computation_types import at_server as type_at_server
from tensorflow_federated.python.core.impl.types.computation_types import FederatedType
from tensorflow_federated.python.core.impl.types.computation_types import FunctionType
from tensorflow_federated.python.core.impl.types.computation_types import SequenceType
from tensorflow_federated.python.core.impl.types.computation_types import StructType
from tensorflow_federated.python.core.impl.types.computation_types import StructWithPythonType
from tensorflow_federated.python.core.impl.types.computation_types import TensorType
from tensorflow_federated.python.core.impl.types.computation_types import to_type
from tensorflow_federated.python.core.impl.types.computation_types import Type
from tensorflow_federated.python.core.impl.types.placements import CLIENTS
from tensorflow_federated.python.core.impl.types.placements import SERVER
from tensorflow_federated.python.core.impl.types.type_conversions import structure_from_tensor_type_tree
from tensorflow_federated.python.core.impl.types.typed_object import TypedObject

if sys.version_info[0] < 3 or sys.version_info[1] < 7:
  raise Exception('TFF only supports Python versions 3.7 or later.')

# Initialize a default execution context. This is implicitly executed the
# first time a module in the `core` package is imported.
backends.native.set_local_python_execution_context()

# Remove packages that are not part of the public API but are picked up due to
# the directory structure. The python import statements above implicitly add
# these to locals().
del python  # pylint:disable=undefined-variable
del proto  # pylint:disable=undefined-variable
