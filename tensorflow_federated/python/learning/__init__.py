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
"""Libraries for using Federated Learning algorithms."""

from tensorflow_federated.python.learning import framework
from tensorflow_federated.python.learning import metrics
from tensorflow_federated.python.learning import models
from tensorflow_federated.python.learning import optimizers
from tensorflow_federated.python.learning import reconstruction
from tensorflow_federated.python.learning.client_weight_lib import ClientWeighting
from tensorflow_federated.python.learning.debug_measurements import add_debug_measurements
from tensorflow_federated.python.learning.federated_averaging import build_federated_averaging_process
from tensorflow_federated.python.learning.federated_averaging import ClientFedAvg
from tensorflow_federated.python.learning.federated_evaluation import build_federated_evaluation
from tensorflow_federated.python.learning.federated_evaluation import build_local_evaluation
from tensorflow_federated.python.learning.federated_sgd import build_federated_sgd_process
from tensorflow_federated.python.learning.framework.optimizer_utils import state_with_new_model_weights
from tensorflow_federated.python.learning.keras_utils import federated_aggregate_keras_metric
from tensorflow_federated.python.learning.keras_utils import from_keras_model
from tensorflow_federated.python.learning.model import BatchOutput
from tensorflow_federated.python.learning.model import Model
from tensorflow_federated.python.learning.model_update_aggregator import compression_aggregator
from tensorflow_federated.python.learning.model_update_aggregator import dp_aggregator
from tensorflow_federated.python.learning.model_update_aggregator import robust_aggregator
from tensorflow_federated.python.learning.model_update_aggregator import secure_aggregator
from tensorflow_federated.python.learning.model_utils import ModelWeights
from tensorflow_federated.python.learning.personalization_eval import build_personalization_eval
