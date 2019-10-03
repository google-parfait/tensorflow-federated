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
"""The public API for contributors who develop federated learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.learning.framework.encoding_utils import build_encoded_broadcast_from_model
from tensorflow_federated.python.learning.framework.encoding_utils import build_encoded_mean_from_model
from tensorflow_federated.python.learning.framework.encoding_utils import build_encoded_sum_from_model
from tensorflow_federated.python.learning.framework.optimizer_utils import build_model_delta_optimizer_process
from tensorflow_federated.python.learning.framework.optimizer_utils import build_stateless_broadcaster
from tensorflow_federated.python.learning.framework.optimizer_utils import ClientDeltaFn
from tensorflow_federated.python.learning.framework.optimizer_utils import ClientOutput
from tensorflow_federated.python.learning.model_utils import enhance
from tensorflow_federated.python.learning.model_utils import EnhancedModel
from tensorflow_federated.python.learning.model_utils import EnhancedTrainableModel
from tensorflow_federated.python.learning.model_utils import ModelWeights

# Used by doc generation script.
_allowed_symbols = [
    "ClientDeltaFn",
    "ClientOutput",
    "EnhancedModel",
    "EnhancedTrainableModel",
    "ModelWeights",
    "build_encoded_broadcast_from_model",
    "build_encoded_mean_from_model",
    "build_encoded_sum_from_model",
    "build_model_delta_optimizer_process",
    "build_stateless_broadcaster",
    "enhance",
]
