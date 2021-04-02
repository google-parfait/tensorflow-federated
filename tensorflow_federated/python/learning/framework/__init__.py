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
"""Libraries for developing Federated Learning algorithms."""

from tensorflow_federated.python.learning.framework.encoding_utils import build_encoded_broadcast_process_from_model
from tensorflow_federated.python.learning.framework.optimizer_utils import build_model_delta_optimizer_process
from tensorflow_federated.python.learning.framework.optimizer_utils import build_stateless_broadcaster
from tensorflow_federated.python.learning.framework.optimizer_utils import build_stateless_mean
from tensorflow_federated.python.learning.framework.optimizer_utils import ClientDeltaFn
from tensorflow_federated.python.learning.framework.optimizer_utils import ClientOutput
from tensorflow_federated.python.learning.framework.optimizer_utils import ServerState
from tensorflow_federated.python.learning.model_utils import enhance
from tensorflow_federated.python.learning.model_utils import EnhancedModel
from tensorflow_federated.python.learning.model_utils import ModelWeights
from tensorflow_federated.python.learning.model_utils import parameter_count_from_model
from tensorflow_federated.python.learning.model_utils import weights_type_from_model
