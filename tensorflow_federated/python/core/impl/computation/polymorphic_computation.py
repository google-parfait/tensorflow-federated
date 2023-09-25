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
"""Utilities for Python functions, defuns, and other types of callables."""

from collections.abc import Callable
from typing import Optional

from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions


class PolymorphicComputation:
  """A generic polymorphic function that accepts arguments of diverse types."""

  def __init__(
      self,
      concrete_function_factory: Callable[
          [computation_types.Type, Optional[bool]],
          computation_impl.ConcreteComputation,
      ],
  ):
    """Crates a polymorphic function with a given function factory.

    Args:
      concrete_function_factory: A callable that accepts a (non-None) TFF type
        as an argument, as well as an optional boolean `unpack` argument which
        should be treated as documented in
        `function_utils.wrap_as_zero_or_one_arg_callable`. The callable must
        return a `Computation` instance that's been created to accept a single
        positional argument of this TFF type (to be reused for future calls with
        parameters of a matching type).
    """
    self._concrete_function_factory = concrete_function_factory
    self._concrete_function_cache = {}

  def fn_for_argument_type(
      self, arg_type: computation_types.Type, unpack: Optional[bool] = None
  ) -> computation_impl.ConcreteComputation:
    """Concretizes this function with the provided `arg_type`.

    The first time this function is called with a particular type on a
    given `PolymorphicComputation` (or this `PolymorphicComputation` is called
    with an argument of the given type), the underlying function will be
    traced using the provided argument type as input. Later calls will
    return the cached computed concrete function.

    Args:
      arg_type: The argument type to use when concretizing this function.
      unpack: Whether to force unpacking the arguments (`True`), never unpack
        the arguments (`False`), or infer whether or not to unpack the arguments
        (`None`).

    Returns:
      The `tff.framework.ConcreteComputation` that results from tracing this
      `PolymorphicComputation` with `arg_type.
    """
    key = repr(arg_type) + str(unpack)
    concrete_fn = self._concrete_function_cache.get(key)
    if not concrete_fn:
      concrete_fn = (self._concrete_function_factory)(arg_type, unpack)
      if concrete_fn.type_signature.parameter != arg_type:
        raise TypeError(
            'Expected a concrete function that takes parameter {}, got one '
            'that takes {}.'.format(
                arg_type, concrete_fn.type_signature.parameter
            )
        )
      self._concrete_function_cache[key] = concrete_fn
    return concrete_fn

  def __call__(self, *args, **kwargs):
    """Invokes this polymorphic function with a given set of arguments.

    Args:
      *args: Positional args.
      **kwargs: Keyword args.

    Returns:
      The result of calling a concrete function, instantiated on demand based
      on the argument types (and cached for future calls).

    Raises:
      TypeError: if the concrete functions created by the factory are of the
        wrong computation_types.
    """
    # TODO: b/113112885 - We may need to normalize individuals args, such that
    # the type is more predictable and uniform (e.g., if someone supplies an
    # unordered dictionary), possibly by converting dict-like and tuple-like
    # containers into `Struct`s.
    packed_arg = function_utils.pack_args_into_struct(args, kwargs)
    arg_type = type_conversions.infer_type(packed_arg)
    # We know the argument types have been packed, so force unpacking.
    concrete_fn = self.fn_for_argument_type(arg_type, unpack=True)
    return concrete_fn(packed_arg)
