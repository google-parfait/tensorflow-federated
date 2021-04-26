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
"""The medium for exchanging executable logic between compiler and runtime."""

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import typed_object


class ComputationModule(typed_object.TypedObject):
  """Wraps around an IREE compiler module created from a TFF computation.

  Instances of this class will be created by helper functions in the `compiler`
  module, and consumed by those in the `runtime` module.
  """

  # TODO(b/153499219): Incorporate into this class information about the
  # binding between TFF type and the ABI of the generated IREE function.
  # This is not needed right now, since at this point, we do not support
  # structured inputs/outputs yet, but it will be needed eventually.

  def __init__(self, compiler_module, function_name, type_signature):
    """Creates an instance of this class.

    Args:
      compiler_module: An instance of `iree_compiler.CompilerModule`.
      function_name: The name of the (possibly only) function in the IREE
        `compiler_module` that represents the entry point of the computation.
      type_signature: The TFF type signature of the function `function_name`,
        always as an instance of `computation_types.FunctionType`, possibly with
        a `None` parameter in case it takes no arguments.
    """
    py_typecheck.check_type(function_name, str)
    py_typecheck.check_type(type_signature, computation_types.FunctionType)
    self._compiler_module = compiler_module
    self._function_name = function_name
    self._type_signature = type_signature

  @property
  def compiler_module(self):
    return self._compiler_module

  @property
  def function_name(self):
    return self._function_name

  @property
  def type_signature(self):
    return self._type_signature
