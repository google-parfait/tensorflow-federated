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
"""Utility classes and functions for integration with MapReduce-like backends.

This module contains utility components for interfacing between TFF and backend
systems that offer MapReduce-like capabilities, i.e., systems that can perform
parallel processing on a set of clients, and then aggregate the results of such
processing on the server. Systems of this type do not support the full
expressiveness of TFF, but they are common enough in practice to warrant a
dedicated set of utility functions, and many examples of TFF computations,
including those constructed by `tff.learning`, can be compiled by TFF into a
form that can be deployed on such systems.
"""

# TODO(b/138261370): Cover this in the general set of guidelines for deployment.

from tensorflow_federated.python.core.backends.mapreduce.canonical_form import CanonicalForm
from tensorflow_federated.python.core.backends.mapreduce.canonical_form_utils import get_canonical_form_for_iterative_process
from tensorflow_federated.python.core.backends.mapreduce.canonical_form_utils import get_iterative_process_for_canonical_form
