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
"""Libraries for running TensorFlow Federated simulations."""

from tensorflow_federated.python.simulation import baselines
from tensorflow_federated.python.simulation import datasets
from tensorflow_federated.python.simulation import models
from tensorflow_federated.python.simulation.checkpoint_manager import FileCheckpointManager
from tensorflow_federated.python.simulation.csv_manager import CSVMetricsManager
from tensorflow_federated.python.simulation.csv_manager import SaveMode
from tensorflow_federated.python.simulation.iterative_process_compositions import compose_dataset_computation_with_computation
from tensorflow_federated.python.simulation.iterative_process_compositions import compose_dataset_computation_with_iterative_process
from tensorflow_federated.python.simulation.metrics_manager import MetricsManager
from tensorflow_federated.python.simulation.sampling_utils import build_uniform_sampling_fn
from tensorflow_federated.python.simulation.server_utils import run_server
from tensorflow_federated.python.simulation.server_utils import server_context
from tensorflow_federated.python.simulation.tensorboard_manager import TensorBoardManager
from tensorflow_federated.python.simulation.training_loop import EVALUATION_METRICS_PREFIX
from tensorflow_federated.python.simulation.training_loop import EVALUATION_TIME_KEY
from tensorflow_federated.python.simulation.training_loop import ROUND_NUMBER_KEY
from tensorflow_federated.python.simulation.training_loop import ROUND_TIME_KEY
from tensorflow_federated.python.simulation.training_loop import run_simulation
from tensorflow_federated.python.simulation.training_loop import run_simulation_with_callbacks
from tensorflow_federated.python.simulation.training_loop import run_stateless_simulation
from tensorflow_federated.python.simulation.training_loop import run_training_process
from tensorflow_federated.python.simulation.training_loop import TRAINING_TIME_KEY
from tensorflow_federated.python.simulation.training_loop import VALIDATION_METRICS_PREFIX
from tensorflow_federated.python.simulation.training_loop import VALIDATION_TIME_KEY
