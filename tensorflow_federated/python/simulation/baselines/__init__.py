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
"""Libraries for constructing baseline learning tasks suitable for simulation."""

from tensorflow_federated.python.simulation.baselines import cifar100
from tensorflow_federated.python.simulation.baselines import emnist
from tensorflow_federated.python.simulation.baselines import shakespeare
from tensorflow_federated.python.simulation.baselines import stackoverflow
from tensorflow_federated.python.simulation.baselines.baseline_task import BaselineTask
from tensorflow_federated.python.simulation.baselines.client_spec import ClientSpec
from tensorflow_federated.python.simulation.baselines.task_data import BaselineTaskDatasets
