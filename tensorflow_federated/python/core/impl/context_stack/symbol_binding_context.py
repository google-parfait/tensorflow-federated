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
"""Defines interface for contexts which can bind symbols."""

import abc
from typing import Any, List, Tuple

from tensorflow_federated.python.core.impl.context_stack import context_base


class SymbolBindingContext(context_base.Context, metaclass=abc.ABCMeta):
  """Interface for contexts which handle binding and tracking of references."""

  @abc.abstractmethod
  def bind_computation_to_reference(self, comp: Any) -> Any:
    """Binds a computation to a symbol, returns a reference to this binding."""
    raise NotImplementedError

  @abc.abstractproperty
  def symbol_bindings(self) -> List[Tuple[str, Any]]:
    """Returns all symbols bound in this context."""
    raise NotImplementedError
