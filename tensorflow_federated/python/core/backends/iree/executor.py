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
"""A substitute for the eager TF executor that delegates to the IREE runtime."""

import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import tracing
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.backends.iree import backend_info
from tensorflow_federated.python.core.backends.iree import compiler
from tensorflow_federated.python.core.backends.iree import runtime
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_utils
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import type_analysis


def to_representation_for_type(value, type_spec=None, backend=None):
  """Verifies or converts the `value` to executor payload matching `type_spec`.

  The following kinds of `value` are supported:

  * Computations, either `pb.Computation` or `computation_impl.ComputationImpl`.
    These are compiled and converted into `runtime.ComputationCallable`.

  * Numpy arrays and scalars, or Python scalars that are converted to Numpy.

  Args:
    value: The raw representation of a value to compare against `type_spec` and
      potentially to be converted.
    type_spec: An instance of `tff.Type`. Can be `None` for values that derive
      from `typed_object.TypedObject`.
    backend: Optional information about the backend, only required for
      computations. Must be `None` or an instance of `backend_info.BackendInfo`.

  Returns:
    Either `value` itself, or a modified version of it.

  Raises:
    TypeError: If the `value` is not compatible with `type_spec`.
    ValueError: If the arguments are incorrect (e.g., missing `backend` for a
      computation-typed `value`).
  """
  type_spec = executor_utils.reconcile_value_with_type_spec(value, type_spec)
  if backend is not None:
    py_typecheck.check_type(backend, backend_info.BackendInfo)
  if isinstance(value, computation_base.Computation):
    return to_representation_for_type(
        computation_impl.ComputationImpl.get_proto(value),
        type_spec=type_spec,
        backend=backend)
  elif isinstance(value, pb.Computation):
    if backend is None:
      raise ValueError('Missing backend info for a computation.')
    module = compiler.import_tensorflow_computation(value)
    return runtime.ComputationCallable(module, backend)
  elif isinstance(type_spec, computation_types.TensorType):
    type_spec.shape.assert_is_fully_defined()
    type_analysis.check_type(value, type_spec)
    if type_spec.shape.rank == 0:
      return np.dtype(type_spec.dtype.as_numpy_dtype).type(value)
    elif type_spec.shape.rank > 0:
      return np.array(value, dtype=type_spec.dtype.as_numpy_dtype)
    else:
      raise TypeError('Unsupported tensor shape {}.'.format(type_spec.shape))
  else:
    raise TypeError('Unexpected type {}.'.format(type_spec))


class IreeValue(executor_value_base.ExecutorValue):
  """A representation of a value managed by the IREE executor."""

  def __init__(self, value, type_spec=None, backend=None):
    """Creates an instance of a value in this executor.

    Args:
      value: Same as in `to_representation_for_type()`.
      type_spec: Same as in `to_representation_for_type()`.
      backend: Same as in `to_representation_for_type()`.
    """
    if type_spec is None:
      py_typecheck.check_type(value, typed_object.TypedObject)
      type_spec = value.type_signature
    else:
      type_spec = computation_types.to_type(type_spec)
      py_typecheck.check_type(type_spec, computation_types.Type)
    self._type_signature = type_spec
    self._value = to_representation_for_type(
        value, type_spec=type_spec, backend=backend)

  @property
  def internal_representation(self):
    """Returns the actual representation of the value embedded in the executor.

    This property is only intended to be used by the executor and tests, not by
    the consumers of the executor interface.
    """
    return self._value

  @property
  def type_signature(self):
    return self._type_signature

  @tracing.trace
  async def compute(self):
    # TODO(b/153499219): Add support for values of other types than tensors.
    py_typecheck.check_type(self._type_signature, computation_types.TensorType)
    return self._value


class IreeExecutor(executor_base.Executor):
  """The IREE executor delegates execution to the IREE runtime.

  NOTE: This executor is in the process of gewtting developed. Most capabilities
  are not implemented, and those that are, only work in experimental mode. All
  aspects of this executor are subject to change.

  Capabilities currently supported in experimental-only mode:

  * Creating TF computations with single-tensor inputs/outputs, and invoking
    them on single-tensor arguments.

  This executor is designed as a drop-in replacement for the eager TF executor.
  It uses a local IREE runtime instead an eager TensorFlow runtime. It will
  support as subset of the capabilities of the eager TF runtime, with the exact
  set of capabilities supported here evolving over time (concurrently with the
  evolution of IREE itself). At this point, the use of this executor is limited
  to incubation and testing of the IREE stack.
  """

  # TODO(b/153499219): Reach full functional parity with the eager TF executor,
  # and beyond, possibly including eventual support for federated compurations
  # that only involve non-federatd (e.g., sequence) operators, and can cleanly
  # map to functions that can be compiled to IREE.

  def __init__(self, backend):
    """Creates a new instance of an IREE executor.

    Args:
      backend: An instance of `backend_info.BackendInfo` to target.
    """
    py_typecheck.check_type(backend, backend_info.BackendInfo)
    self._backend_info = backend

  @tracing.trace(span=True)
  async def create_value(self, value, type_spec=None):
    """Embeds `value` of type `type_spec` within this executor.

    Args:
      value: An object that represents the value to embed within the executor.
      type_spec: The `tff.Type` of the value represented by this object, or
        something convertible to it. Can optionally be `None` if `value` is an
        instance of `typed_object.TypedObject`.

    Returns:
      An instance of `IreeValue`.

    Raises:
      TypeError: If the arguments are of the wrong types.
      ValueError: If the type was not specified and cannot be determined from
        the value.
    """
    return IreeValue(value, type_spec, self._backend_info)

  @tracing.trace
  async def create_call(self, comp, arg=None):
    py_typecheck.check_type(comp, IreeValue)
    if arg is not None:
      py_typecheck.check_type(arg, IreeValue)
    py_typecheck.check_type(comp.type_signature, computation_types.FunctionType)
    py_typecheck.check_callable(comp.internal_representation)
    if comp.type_signature.parameter is not None:
      result_struct = comp.internal_representation(
          parameter=arg.internal_representation)
    else:
      result_struct = comp.internal_representation()

    return IreeValue(result_struct['result'], comp.type_signature.result,
                     self._backend_info)

  # TODO(b/153499219): Implement tuples and selections below.

  @tracing.trace
  async def create_struct(self, elements):
    raise NotImplementedError

  @tracing.trace
  async def create_selection(self, source, index=None, name=None):
    raise NotImplementedError

  def close(self):
    pass
