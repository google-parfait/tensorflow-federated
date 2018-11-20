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
"""TensorFlow Federated library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We are doing a wildcard import here, since symbols to export have already been
# explicitly whitelisted in core/api, and it makes no sense to repeat them here.
# pylint: disable=wildcard-import
from tensorflow_federated.python.core.api import *


# Used by doc generation script.
_allowed_symbols = [
    "CLIENTS",
    "Computation",
    "FederatedType",
    "FunctionType",
    "NamedTupleType",
    "SERVER",
    "SequenceType",
    "TensorType",
    "Type",
    "federated_computation",
    "tf_computation",
    "to_type",
]
