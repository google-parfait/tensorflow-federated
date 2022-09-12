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
"""Libraries of specialized processes used for building learning algorithms."""

from tensorflow_federated.python.learning.templates.apply_optimizer_finalizer import build_apply_optimizer_finalizer
from tensorflow_federated.python.learning.templates.client_works import ClientResult
from tensorflow_federated.python.learning.templates.client_works import ClientWorkProcess
from tensorflow_federated.python.learning.templates.composers import compose_learning_process
from tensorflow_federated.python.learning.templates.composers import LearningAlgorithmState
from tensorflow_federated.python.learning.templates.distributors import build_broadcast_process
from tensorflow_federated.python.learning.templates.distributors import DistributionProcess
from tensorflow_federated.python.learning.templates.finalizers import FinalizerProcess
from tensorflow_federated.python.learning.templates.learning_process import LearningProcess
from tensorflow_federated.python.learning.templates.learning_process import LearningProcessOutput
from tensorflow_federated.python.learning.templates.model_delta_client_work import build_model_delta_client_work
