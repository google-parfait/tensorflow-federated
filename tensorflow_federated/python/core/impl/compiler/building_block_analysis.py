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
"""A library of static analysis functions for building blocks."""

from tensorflow_federated.python.core.impl.compiler import building_blocks


def is_called_intrinsic(comp, uri=None):
  """Tests if `comp` is a called intrinsic with the given `uri`.

  Args:
    comp: The computation building block to test.
    uri: An optional URI or list of URIs; the same form as what is accepted by
      isinstance.

  Returns:
    `True` if `comp` is a called intrinsic with the given `uri`, otherwise
    `False`.
  """
  if isinstance(uri, str):
    uri = [uri]
  return (
      isinstance(comp, building_blocks.Call)
      and isinstance(comp.function, building_blocks.Intrinsic)
      and (uri is None or comp.function.uri in uri)
  )


def is_identity_function(comp):
  """Returns `True` if `comp` is an identity function, otherwise `False`."""
  return (
      isinstance(comp, building_blocks.Lambda)
      and isinstance(comp.result, building_blocks.Reference)
      and comp.parameter_name == comp.result.name
  )
