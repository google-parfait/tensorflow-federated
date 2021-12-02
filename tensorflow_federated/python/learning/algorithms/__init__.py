# Copyright 2021, The TensorFlow Federated Authors.
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
"""Libraries providing implementations of federated learning algorithms."""

from tensorflow_federated.python.learning.algorithms.client_scheduled_federated_averaging import build_client_scheduled_federated_averaging_process
from tensorflow_federated.python.learning.algorithms.fed_prox import build_example_weighted_fed_prox_process
