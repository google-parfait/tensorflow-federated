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
"""Defines functions and classes for building and manipulating TFF values."""

from tensorflow_federated.python.core.impl import value_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl

# TODO(b/113116813): Pick accepted representations for sequence and federated
# type constants and update this as well as value_impl.ValueImpl accordingly.


def to_value(val, type_spec=None):
  """Converts the argument into an instance of the abstract class `tff.Value`.

  Instances of `tff.Value` represent TFF values that appear internally in
  federated computations. This helper function can be used to wrap a variety of
  Python objects as `tff.Value` instances to allow them to be passed as
  arguments, used as functions, or otherwise manipulated within bodies of
  federated computations.

  At the moment, the supported types include:

  * Simple constants of `str`, `int`, `float`, and `bool` types, mapped to
    values of a TFF tensor type.

  * Numpy arrays (`np.ndarray` objects), also mapped to TFF tensors.

  * Dictionaries (`collections.OrderedDict` and unordered `dict`), `list`s,
    `tuple`s, `namedtuple`s, and `Struct`s, all of which are mapped to
    TFF tuple type.

  * Computations (constructed with either the `tff.tf_computation` or with the
    `tff.federated_computation` decorator), typically mapped to TFF functions.

  * Placement literals (`tff.CLIENTS`, `tff.SERVER`), mapped to values of the
    TFF placement type.

  This function is also invoked when attempting to execute a TFF computation.
  All arguments supplied in the invocation are converted into TFF values prior
  to execution. The types of Python objects that can be passed as arguments to
  computations thus matches the types listed here.

  Args:
    val: An instance of one of the Python types that are convertible to TFF
      values (instances of `tff.Value`).
    type_spec: An optional type specifier that allows for disambiguating the
      target type (e.g., when two TFF types can be mapped to the same Python
      representations). If not specified, TFF tried to determine the type of the
      TFF value automatically.

  Returns:
    An instance of `tff.Value` of a TFF type as described above.
  """
  return value_impl.to_value(val, type_spec, context_stack_impl.context_stack)
