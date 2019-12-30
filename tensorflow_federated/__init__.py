# Lint as: python3
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
"""TensorFlow Federated Library."""

import sys
import warnings

from tensorflow_federated.version import __version__  # pylint: disable=g-bad-import-order

from tensorflow_federated.python.core.api.computation_base import Computation
from tensorflow_federated.python.core.api.computation_types import FederatedType
from tensorflow_federated.python.core.api.computation_types import FunctionType
from tensorflow_federated.python.core.api.computation_types import NamedTupleType
from tensorflow_federated.python.core.api.computation_types import SequenceType
from tensorflow_federated.python.core.api.computation_types import TensorType
from tensorflow_federated.python.core.api.computation_types import to_type
from tensorflow_federated.python.core.api.computation_types import Type
from tensorflow_federated.python.core.api.computations import federated_computation
from tensorflow_federated.python.core.api.computations import tf_computation
from tensorflow_federated.python.core.api.intrinsics import federated_aggregate
from tensorflow_federated.python.core.api.intrinsics import federated_apply
from tensorflow_federated.python.core.api.intrinsics import federated_broadcast
from tensorflow_federated.python.core.api.intrinsics import federated_collect
from tensorflow_federated.python.core.api.intrinsics import federated_map
from tensorflow_federated.python.core.api.intrinsics import federated_mean
from tensorflow_federated.python.core.api.intrinsics import federated_reduce
from tensorflow_federated.python.core.api.intrinsics import federated_sum
from tensorflow_federated.python.core.api.intrinsics import federated_value
from tensorflow_federated.python.core.api.intrinsics import federated_zip
from tensorflow_federated.python.core.api.intrinsics import sequence_map
from tensorflow_federated.python.core.api.intrinsics import sequence_reduce
from tensorflow_federated.python.core.api.intrinsics import sequence_sum
from tensorflow_federated.python.core.api.placements import CLIENTS
from tensorflow_federated.python.core.api.placements import SERVER
from tensorflow_federated.python.core.api.typed_object import TypedObject
from tensorflow_federated.python.core.api.value_base import Value
from tensorflow_federated.python.core.api.values import to_value

# NOTE: These imports must happen after the API imports.
# pylint: disable=g-bad-import-order
from tensorflow_federated.python.core import framework
from tensorflow_federated.python.core import backends
from tensorflow_federated.python.core import utils
# pylint: enable=g-bad-import-order

# NOTE: These imports must happen after the Core imports.
# pylint: disable=g-bad-import-order
from tensorflow_federated.python import learning
from tensorflow_federated.python import simulation
# pylint: enable=g-bad-import-order,wildcard-import


# pylint: disable=g-bad-exception-name
class Python2DeprecationWarning(Warning):
  pass


# pylint: enable=g-bad-exception-name

if sys.version_info[0] < 3:
  warnings.warn(
      "Using TensorFlow Federated with Python 2 is deprecated and will be removed in January 2020.\n"
      "See https://python3statement.org/ for more information.",
      Python2DeprecationWarning)

# Used by doc generation script.
_allowed_symbols = [
    "backends",
    "CLIENTS",
    "Computation",
    "FederatedType",
    "FunctionType",
    "NamedTupleType",
    "SERVER",
    "SequenceType",
    "TensorType",
    "Type",
    "TypedObject",
    "Value",
    "federated_aggregate",
    "federated_apply",
    "federated_broadcast",
    "federated_collect",
    "federated_computation",
    "federated_map",
    "federated_mean",
    "federated_reduce",
    "federated_sum",
    "federated_value",
    "federated_zip",
    "framework",
    "learning",
    "sequence_map",
    "sequence_reduce",
    "sequence_sum",
    "simulation",
    "tf_computation",
    "to_type",
    "to_value",
    "utils",
]
