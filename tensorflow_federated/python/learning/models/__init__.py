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
"""Libraries for working with models in federated learning algorithms."""

from tensorflow_federated.python.learning.models.functional import functional_model_from_keras
from tensorflow_federated.python.learning.models.functional import FunctionalModel
from tensorflow_federated.python.learning.models.functional import model_from_functional
from tensorflow_federated.python.learning.models.serialization import load
from tensorflow_federated.python.learning.models.serialization import load_functional_model
from tensorflow_federated.python.learning.models.serialization import save
from tensorflow_federated.python.learning.models.serialization import save_functional_model
