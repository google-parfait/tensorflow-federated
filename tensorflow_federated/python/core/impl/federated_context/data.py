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
"""The TFF data construct."""

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.types import computation_types


def data(uri: str, type_spec: computation_types.Type):
  """Constructs a TFF `data` computation with the given URI and TFF type.

  Args:
    uri: A string (`str`) URI of the data.
    type_spec: An instance of `tff.Type` that represents the type of this data.

  Returns:
    A representation of the data with the given URI and TFF type in the body of
    a federated computation.

  Raises:
    TypeError: If the arguments are not of the types specified above.
  """
  py_typecheck.check_type(uri, str)
  type_spec = computation_types.to_type(type_spec)
  return value_impl.to_value(building_blocks.Data(uri, type_spec), type_spec)
