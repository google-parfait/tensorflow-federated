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
"""Utility classes/functions built on top of TensorFlow Federated Core API.

All components that depend on utils should import symbols from this file rather
than directly importing individual modules. For this reason, the visibility for
the latter is set to private and should remain such. The code in utils must not
depend on implementation classes. It should be written against the Core API.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.core.utils.computation_utils import IterativeProcess
from tensorflow_federated.python.core.utils.computation_utils import StatefulAggregateFn
from tensorflow_federated.python.core.utils.computation_utils import StatefulBroadcastFn
from tensorflow_federated.python.core.utils.computation_utils import update_state
from tensorflow_federated.python.core.utils.differential_privacy import build_dp_aggregate
from tensorflow_federated.python.core.utils.encoding_utils import build_encoded_broadcast
from tensorflow_federated.python.core.utils.encoding_utils import build_encoded_mean
from tensorflow_federated.python.core.utils.encoding_utils import build_encoded_sum
from tensorflow_federated.python.core.utils.federated_aggregations import federated_max
from tensorflow_federated.python.core.utils.federated_aggregations import federated_min
from tensorflow_federated.python.core.utils.tf_computation_utils import assign
from tensorflow_federated.python.core.utils.tf_computation_utils import create_variables
from tensorflow_federated.python.core.utils.tf_computation_utils import identity

# Used by doc generation script.
_allowed_symbols = [
    "StatefulAggregateFn",
    "StatefulBroadcastFn",
    "IterativeProcess",
    "assign",
    "build_dp_aggregate",
    "build_encoded_broadcast",
    "build_encoded_mean",
    "build_encoded_sum",
    "federated_max",
    "federated_min",
    "create_variables",
    "identity",
    "update_state",
]
