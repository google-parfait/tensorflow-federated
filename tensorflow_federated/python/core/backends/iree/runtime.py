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

import iree.compiler
import iree.runtime

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.backends.iree import backend_info
from tensorflow_federated.python.core.backends.iree import computation_module
from tensorflow_federated.python.core.impl.types import typed_object

# A mutex that protects `_driver_name_to_config_dict`.
_driver_name_to_config_lock = threading.Lock()

# A mapping from driver names to `iree.runtime.Config` instances.
_driver_name_to_config_dict = {}


def _get_default_config_for_driver(driver_name):
  """Returns an IREE runtime config for the given driver.

  Enforces that there is always at most one config per driver.

  Args:
    driver_name: A string that represents the name of the driver.

  Returns:
    An instance of `iree.runtime.Config` for this driver.
  """
  # TODO(b/153499219): Upstream this to IREE in some form (we won't block on
  # this here, though, since upstreaming well would mean yanking the constructor
  # for `iree.runtime.Config` and updating all existing call sites that use it).
  py_typecheck.check_type(driver_name, str)
  with _driver_name_to_config_lock:
    config = _driver_name_to_config_dict.get(driver_name)
    if config is None:
      config = iree.runtime.Config(driver_name=driver_name)
      _driver_name_to_config_dict[driver_name] = config
    return config


class ComputationCallable(typed_object.TypedObject):
  """An internal callable that encapsulates the logic of a single computation.

  NOTE: The exact structure and implementation of this class may change, or the
  class may be removed or replaced with something else.
  """

  def __init__(self, module, backend):
    """Creates this callable for a given computation moduel and backend.

    Args:
      module: An instance of `computation_module.ComputationModule`.
      backend: An instance of `backend_info.BackendInfo`.
    """
    py_typecheck.check_type(module, computation_module.ComputationModule)
    py_typecheck.check_type(backend, backend_info.BackendInfo)
    flatbuffer_blob = iree.compiler.compile_str(
        module.compiler_module, target_backends=[backend.target_name])
    # TODO(b/153499219): Find a way to name the modules somehow differently
    # for debugging. Right now, module names come from the implicit "module {}"
    # that wraps anything parsed from ASM that lacks an explicit module
    # declaration. Possibly manually surround with "module @myName { ... }" in
    # the compiler helpers.
    self._vm_module = iree.runtime.VmModule.from_flatbuffer(flatbuffer_blob)
    self._config = _get_default_config_for_driver(backend.driver_name)
    self._function_name = module.function_name
    self._type_signature = module.type_signature

  @property
  def type_signature(self):
    return self._type_signature

  def __call__(self, *args, **kwargs):
    """Invokes this callable with the given set of arguments.

    Args:
      *args: Positional arguments.
      **kwargs: Keyword arguments.

    Returns:
      The result of the call.
    """
    # Context creation can be expected to be on the order of milliseconds or
    # less, so constructing one per call should be cheap enough, and can make
    # things simpler while we look for ways to support true local variables
    # in IREE and eliminate any kind of global state.
    context = iree.runtime.SystemContext(config=self._config)
    context.add_module(self._vm_module)
    callable_fn = getattr(context.modules.module, self._function_name)
    return callable_fn(*args, **kwargs)


def compile_and_run_on_args(module, backend, *args, **kwargs):
  """Helper that compiles runs a given compiled module with a given set of args.

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
  # NOTE: Even though it's a one-liner, we want it as a way to isolate compiler
  # tests from the internal details of how the runtime is plumbed together.
  return ComputationCallable(module, backend)(*args, **kwargs)
