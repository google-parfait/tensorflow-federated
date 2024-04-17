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
"""Defines context interfaces which evaluates computation invocations.

Invocations of TensorFlow Federated computations need to be treated differently
depending on the context in which they are invoked. For example:

*   During top-level Python simulations, computation invocations result in the
    computation being serialized and evaluated by the TensorFlow native runtime.

*   In functions decorated with `@tff.tensorflow.computation`, computation
    invocations must import the body of the invoked function into the current
    TensorFlow graph.

Code can customize the way in which each of these calls are evaluated by setting
a specific context using a global or thread-local context stack.
"""

import abc
from typing import Any


class ContextError(RuntimeError):
  pass


class SyncContext(metaclass=abc.ABCMeta):
  """A synchronous context to evaluate of computations."""

  @abc.abstractmethod
  def invoke(self, comp: Any, arg: Any) -> Any:
    """Invokes computation `comp` with argument `arg`.

    Args:
      comp: The computation being invoked. The Python type of `comp` expected
        here (e.g., `pb.Computation`. `tff.framework.ConcreteComputation`, or
        other) may depend on the context. It is the responsibility of the
        concrete implementation of this interface to verify that the type of
        `comp` matches what the context is expecting.
      arg: The argument passed to the computation. If no argument is passed,
        this will be `None`. Structural argument types will be normalized into
        `tff.structure.Struct`s.

    Returns:
      The result of invocation, which is context-dependent.
    """
    raise NotImplementedError


class AsyncContext(metaclass=abc.ABCMeta):
  """An asynchronous context to evaluate of computations."""

  @abc.abstractmethod
  async def invoke(self, comp: Any, arg: Any) -> Any:
    """Invokes computation `comp` with argument `arg`.

    Args:
      comp: The computation being invoked. The Python type of `comp` expected
        here (e.g., `pb.Computation`. `tff.framework.ConcreteComputation`, or
        other) may depend on the context. It is the responsibility of the
        concrete implementation of this interface to verify that the type of
        `comp` matches what the context is expecting.
      arg: The argument passed to the computation. If no argument is passed,
        this will be `None`. Structural argument types will be normalized into
        `tff.structure.Struct`s.

    Returns:
      The result of invocation, which is context-dependent.
    """
    raise NotImplementedError
