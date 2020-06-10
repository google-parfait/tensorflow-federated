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
"""A collection of utilities for TFF to interact with the local IREE runtime."""

import threading

from iree.bindings.python.pyiree import rt as iree_runtime
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.backends.iree import backend_info
from tensorflow_federated.python.core.backends.iree import computation_module

# A mutex that protects `_driver_name_to_config_dict`.
_driver_name_to_config_lock = threading.Lock()

# A mapping from driver names to `iree_runtime.Config` instances.
_driver_name_to_config_dict = {}


def _get_default_config_for_driver(driver_name):
  """Returns an IREE runtime config for the given driver.

  Enforces that there is always at most one config per driver.

  Args:
    driver_name: A string that represents the name of the driver.

  Returns:
    An instance of `iree_runtime.Config` for this driver.
  """
  # TODO(b/153499219): Upstream this to IREE in some form (we won't block on
  # this here, though, since upstreaming well would mean yanking the constructor
  # for `iree_runtime.Config` and updating all existing call sites that use it).
  py_typecheck.check_type(driver_name, str)
  with _driver_name_to_config_lock:
    config = _driver_name_to_config_dict.get(driver_name)
    if config is None:
      config = iree_runtime.Config(driver_name=driver_name)
      _driver_name_to_config_dict[driver_name] = config
    return config


def compile_and_run_on_args(module, backend, *args, **kwargs):
  """Helper that compiles runs a given compiled module with a given set of args.

  This helper constructs a separate runtime context, compiles and loads the
  module into it, invokes it on the given arguments, and disposes of resources
  if created to support the invocation.

  NOTE: The intended primary purpose of this helper is testing and debugging.
  The overhead of compilation and loading, and constructing a separate
  throwaway context for each invocation, could be too large for high-performance
  applications.

  Args:
    module: An instance of `computation_module.ComputationModule`.
    backend: An instance of `backend_info.BackendInfo`.
    *args: Positional arguments for invocation.
    **kwargs: Keyword arguments for invocation.

  Returns:
    The result of invocation on IREE.
  """
  py_typecheck.check_type(module, computation_module.ComputationModule)
  py_typecheck.check_type(backend, backend_info.BackendInfo)
  flatbuffer_blob = module.compiler_module.compile(
      target_backends=[backend.target_name])
  vm_module = iree_runtime.VmModule.from_flatbuffer(flatbuffer_blob)
  context = iree_runtime.SystemContext(
      config=_get_default_config_for_driver(backend.driver_name))
  context.add_module(vm_module)
  function_name = module.function_name
  # TODO(b/153499219): Can we possibly name the modules somehow differently?
  # This may not matter if we spawn a separate context for each, but it will
  # matter eventually. Right now, this comes from the implicit "module {}"
  # that wraps anything parsed from ASM that lacks an explicit module
  # declaration. Possibly manually surround with "module @myName { ... }" in
  # the compiler helpers.
  callable_fn = getattr(context.modules.module, function_name)
  return callable_fn(*args, **kwargs)
