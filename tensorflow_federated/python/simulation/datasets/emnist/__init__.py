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
"""Module for the federated EMNIST experimental dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.simulation.datasets.emnist.load_data import get_infinite
from tensorflow_federated.python.simulation.datasets.emnist.load_data import get_synthetic
from tensorflow_federated.python.simulation.datasets.emnist.load_data import load_data


# Used by doc generation script.
_allowed_symbols = [
    "get_infinite",
    "get_synthetic",
    "load_data",
]
