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
"""Libraries for using Federated Reconstruction algorithms."""

from tensorflow_federated.python.learning.reconstruction.evaluation_computation import build_federated_evaluation
from tensorflow_federated.python.learning.reconstruction.keras_utils import from_keras_model
from tensorflow_federated.python.learning.reconstruction.model import BatchOutput
from tensorflow_federated.python.learning.reconstruction.model import Model
from tensorflow_federated.python.learning.reconstruction.reconstruction_utils import build_dataset_split_fn
from tensorflow_federated.python.learning.reconstruction.reconstruction_utils import DatasetSplitFn
from tensorflow_federated.python.learning.reconstruction.reconstruction_utils import get_global_variables
from tensorflow_federated.python.learning.reconstruction.reconstruction_utils import get_local_variables
from tensorflow_federated.python.learning.reconstruction.reconstruction_utils import simple_dataset_split_fn
from tensorflow_federated.python.learning.reconstruction.training_process import build_training_process
from tensorflow_federated.python.learning.reconstruction.training_process import ClientOutput
