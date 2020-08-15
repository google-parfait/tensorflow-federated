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
"""Libraries for running Federated Learning simulations."""

from tensorflow_federated.python.simulation import datasets
from tensorflow_federated.python.simulation import models
from tensorflow_federated.python.simulation.client_data import ClientData
from tensorflow_federated.python.simulation.file_per_user_client_data import FilePerUserClientData
from tensorflow_federated.python.simulation.from_tensor_slices_client_data import FromTensorSlicesClientData
from tensorflow_federated.python.simulation.hdf5_client_data import HDF5ClientData
from tensorflow_federated.python.simulation.iterative_process_compositions import compose_dataset_computation
from tensorflow_federated.python.simulation.server_utils import run_server
from tensorflow_federated.python.simulation.transforming_client_data import TransformingClientData
