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
"""Libraries for developing federated learning algorithms."""

from tensorflow_federated.python.common_libs import deprecation
from tensorflow_federated.python.learning.framework.optimizer_utils import build_model_delta_optimizer_process
from tensorflow_federated.python.learning.framework.optimizer_utils import build_stateless_broadcaster
from tensorflow_federated.python.learning.framework.optimizer_utils import ClientDeltaFn
from tensorflow_federated.python.learning.framework.optimizer_utils import ClientOutput
from tensorflow_federated.python.learning.framework.optimizer_utils import ServerState
from tensorflow_federated.python.learning.models.model_weights import ModelWeights
from tensorflow_federated.python.learning.models.model_weights import weights_type_from_model

weights_type_from_model = deprecation.deprecated(
    weights_type_from_model,
    '`tff.learning.framework.weights_type_from_model` is deprecated, use '
    '`tff.learning.models.weights_type_from_model`.')
