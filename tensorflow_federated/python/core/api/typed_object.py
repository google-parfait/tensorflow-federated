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
"""Defines an abstract interface for things that possess TFF type signatures."""

import abc


class TypedObject(object, metaclass=abc.ABCMeta):
  """An abstract interface for things that possess TFF type signatures."""

  @abc.abstractproperty
  def type_signature(self):
    """Returns the TFF type of this object (an instance of `tff.Type`)."""
    raise NotImplementedError
