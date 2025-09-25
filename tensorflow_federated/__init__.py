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
"""The TensorFlow Federated library."""

import sys

# pylint: disable=g-bad-import-order,g-importing-member
from tensorflow_federated.python import aggregators
from tensorflow_federated.python import analytics
from tensorflow_federated.python import learning
from tensorflow_federated.python import program
from tensorflow_federated.python import simulation
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core import backends
from tensorflow_federated.python.core import framework
from tensorflow_federated.python.core import templates
from tensorflow_federated.python.core.environments import tensorflow
from tensorflow_federated.version import __version__
# pylint: enable=g-bad-import-order,g-importing-member

if sys.version_info < (3, 9):
  raise RuntimeError('TFF only supports Python versions 3.9 or later.')

# Initialize a default execution context. This is implicitly executed the
# first time a module in the `core` package is imported.
backends.native.set_sync_local_cpp_execution_context()

# Remove packages that are not part of the public API but are picked up due to
# the directory structure. The python import statements above implicitly add
# these to locals().
del python  # pylint: disable=undefined-variable
