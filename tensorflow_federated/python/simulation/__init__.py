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
"""The public API for experimenters running federated learning simulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow_federated.python.simulation import datasets
from tensorflow_federated.python.simulation import models
from tensorflow_federated.python.simulation.client_data import ClientData
from tensorflow_federated.python.simulation.file_per_user_client_data import FilePerUserClientData
from tensorflow_federated.python.simulation.from_tensor_slices_client_data import FromTensorSlicesClientData
from tensorflow_federated.python.simulation.hdf5_client_data import HDF5ClientData
from tensorflow_federated.python.simulation.transforming_client_data import TransformingClientData

# High-performance simulation components currently only available in Python 3,
# and dependent on targets are are not currently included in the open-source
# build rule.
# TODO(b/134543154): Modify the OSS build rule to conditionally include these
# new targets if possible.
if six.PY3:
  # pylint: disable=g-import-not-at-top,undefined-variable
  try:
    from tensorflow_federated.python.simulation.server_utils import run_server
  except ModuleNotFoundError:
    pass
  # pylint: enable=g-import-not-at-top,undefined-variable

# Used by doc generation script.
_allowed_symbols = [
    "ClientData",
    "FilePerUserClientData",
    "FromTensorSlicesClientData",
    "HDF5ClientData",
    "TransformingClientData",
    "datasets",
    "models",
    "run_server",
]
