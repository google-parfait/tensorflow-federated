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
"""Defines the abstract interfaces for representations of various TFF values."""

import abc

from tensorflow_federated.python.core.api import typed_object

# TODO(b/113112108): Add derived interfaces and functions to support reflection.


class Value(typed_object.TypedObject, metaclass=abc.ABCMeta):
  """An abstract base class for all values in the bodies of TFF computations.

  This interface is only relevant in the context of non-TensorFlow computations,
  such as those that represent federated orchestration logic. The bodies of
  such computations will contain a mixture of federated communication operators,
  and calls to TensorFlow computations embedded in them as subcomponents. All
  values that appear in those computations implement this common interface, just
  like all values in TensorFlow computations appear as tensors.

  Outside of the bodies of composite non-TensorFlow computations, this interface
  is not used. All fully constructed computations implement `tff.Computation`.
  """

  @abc.abstractmethod
  def __repr__(self):
    """Returns a full-form representation of this value."""
    raise NotImplementedError

  @abc.abstractmethod
  def __str__(self):
    """Returns a concise representation of this value."""
    raise NotImplementedError

  @abc.abstractmethod
  def __dir__(self):
    """For values of a named tuple type, returns the list of named members."""
    raise NotImplementedError

  @abc.abstractmethod
  def __getattr__(self, name):
    """For values of a named tuple type, returns the element named `name`."""
    raise NotImplementedError

  @abc.abstractmethod
  def __len__(self):
    """For values of a named tuple type, returns the number of elements."""
    raise NotImplementedError

  @abc.abstractmethod
  def __getitem__(self, index):
    """For values of a named tuple type, returns the element at `index`."""
    raise NotImplementedError

  @abc.abstractmethod
  def __iter__(self):
    """For values of a named tuple type, iterates over the tuple elements."""
    raise NotImplementedError

  @abc.abstractmethod
  def __call__(self, *args, **kwargs):
    """For values of functional types, invokes this value on given arguments."""
    raise NotImplementedError
