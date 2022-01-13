# Copyright 2019, The TensorFlow Federated Authors.
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
"""Datasets for running TensorFlow Federated simulations."""

from tensorflow_federated.python.simulation.datasets import celeba
from tensorflow_federated.python.simulation.datasets import cifar100
from tensorflow_federated.python.simulation.datasets import emnist
from tensorflow_federated.python.simulation.datasets import gldv2
from tensorflow_federated.python.simulation.datasets import inaturalist
from tensorflow_federated.python.simulation.datasets import shakespeare
from tensorflow_federated.python.simulation.datasets import stackoverflow
from tensorflow_federated.python.simulation.datasets.client_data import ClientData
from tensorflow_federated.python.simulation.datasets.dataset_utils import build_dataset_mixture
from tensorflow_federated.python.simulation.datasets.dataset_utils import build_single_label_dataset
from tensorflow_federated.python.simulation.datasets.dataset_utils import build_synthethic_iid_datasets
from tensorflow_federated.python.simulation.datasets.file_per_user_client_data import FilePerUserClientData
from tensorflow_federated.python.simulation.datasets.from_tensor_slices_client_data import TestClientData
from tensorflow_federated.python.simulation.datasets.sql_client_data import SqlClientData
from tensorflow_federated.python.simulation.datasets.sql_client_data_utils import load_and_parse_sql_client_data
from tensorflow_federated.python.simulation.datasets.sql_client_data_utils import save_to_sql_client_data
from tensorflow_federated.python.simulation.datasets.transforming_client_data import TransformingClientData
