# Copyright 2020, The TensorFlow Federated Authors.
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
"""Library of compiler functions for usage in the native execution context."""

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.impl import do_not_use_compiler


def transform_to_native_form(
    comp: computation_base.Computation) -> computation_base.Computation:
  """Compiles a computation for execution in the TFF native runtime.

  This function transforms the proto underlying `comp` by first transforming it
  to call-dominant form (see `tff.framework.transform_to_call_dominant` for
  definition), then computing information on the dependency structure of the
  bindings and remapping them into tuples, such that every computation is
  evaluated as early as possible, and parallelized with any other computation
  with which it shares dependency structure.

  Args:
    comp: Instance of `computation_base.Computation` to compile.

  Returns:
    A new `computation_base.Computation` representing the compiled version of
    `comp`.
  """

  return do_not_use_compiler._do_not_use_transform_to_native_form(comp)  # pylint: disable=protected-access
