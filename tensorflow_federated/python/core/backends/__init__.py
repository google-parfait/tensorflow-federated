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
"""Backends for constructing, compiling, and executing computations.

Computations expressed in TFF can be executed on a variety of backends,
including native backends that implement TFF's interfaces such as
`tff.framework.Executor`, as well as custom non-native backends such as
MapReduce-like systems that may only be able to run a subset of computations
expressibe in TFF.
"""

from tensorflow_federated.python.core.backends import mapreduce
from tensorflow_federated.python.core.backends import native
from tensorflow_federated.python.core.backends import reference
from tensorflow_federated.python.core.backends import test
