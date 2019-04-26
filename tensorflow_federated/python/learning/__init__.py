# Lint as: python2
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
"""The public API for model developers using federated learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.learning import framework
from tensorflow_federated.python.learning.federated_averaging import build_federated_averaging_process
from tensorflow_federated.python.learning.federated_evaluation import build_federated_evaluation
from tensorflow_federated.python.learning.federated_sgd import build_federated_sgd_process
from tensorflow_federated.python.learning.framework.optimizer_utils import state_with_new_model_weights
from tensorflow_federated.python.learning.model import BatchOutput
from tensorflow_federated.python.learning.model import Model
from tensorflow_federated.python.learning.model import TrainableModel
from tensorflow_federated.python.learning.model_utils import assign_weights_to_keras_model
from tensorflow_federated.python.learning.model_utils import from_compiled_keras_model
from tensorflow_federated.python.learning.model_utils import from_keras_model

# Used by doc generation script.
_allowed_symbols = [
    "BatchOutput",
    "Model",
    "TrainableModel",
    "assign_weights_to_keras_model",
    "build_federated_averaging_process",
    "build_federated_evaluation",
    "build_federated_sgd_process",
    "framework",
    "from_compiled_keras_model",
    "from_keras_model",
    "state_with_new_model_weights",
]
