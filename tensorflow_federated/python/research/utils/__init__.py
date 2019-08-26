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
"""Utility functions used in the research/ directory.

General utilities used by the other directories under `research/`, for things
like writing output, constructing grids of experiments, configuration via
command-line flags, etc.

These utilities are not part of the TFF pip package.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.research.utils.utils_impl import atomic_write_to_csv
from tensorflow_federated.python.research.utils.utils_impl import define_optimizer_flags
from tensorflow_federated.python.research.utils.utils_impl import get_optimizer_from_flags
from tensorflow_federated.python.research.utils.utils_impl import iter_grid
from tensorflow_federated.python.research.utils.utils_impl import record_new_flags

# Used by doc generation script.
_allowed_symbols = [
    "iter_grid",
    "atomic_write_to_csv",
    "define_optimizer_flags",
    "get_optimizer_from_flags",
    "record_new_flags",
]
