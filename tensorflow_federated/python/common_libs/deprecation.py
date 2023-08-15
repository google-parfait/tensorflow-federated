## Copyright 2022, The TensorFlow Federated Authors.
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
"""A module for utilities to notify users of deprecated APIs."""

from collections.abc import Callable
import functools
from typing import Optional, TypeVar
import warnings

_T = TypeVar('_T')


# TODO: b/269491402 - Delete this decorator and use
# `typing_extensions.deprecated`, when it is available.
def deprecated(
    msg: str,
    *,
    category: Optional[type[Warning]] = DeprecationWarning,
    stacklevel: int = 1,
) -> Callable[[_T], _T]:
  """Indicate that a class, function or overload is deprecated.

  Usage:

  ```
  @deprecated("Use B instead")
  class A:
      pass

  @deprecated("Use g instead")
  def f():
      pass

  @overload
  @deprecated("int support is deprecated")
  def g(x: int) -> int: ...

  @overload
  def g(x: str) -> int: ...
  ```

  When this decorator is applied to an object, the type checker
  will generate a diagnostic on usage of the deprecated object.

  No runtime warning is issued. The decorator sets the ``__deprecated__``
  attribute on the decorated object to the deprecation message
  passed to the decorator. If applied to an overload, the decorator
  must be after the ``@overload`` decorator for the attribute to
  exist on the overload as returned by ``get_overloads()``.

  See PEP 702 for details.

  Args:
    msg: The deprecation message.
    category: A warning class. Defaults to `DeprecationWarning`. If this is set
      to `None`, no warning is issued at runtime and the decorator returns the
      original object, except for setting the `__deprecated__` attribute.
    stacklevel: The number of stack frames to skip when issuing the warning.
      Defaults to 1, indicating that the warning should be issued at the site
      where the deprecated object is called. Internally, the implementation will
      add the number of stack frames it uses in wrapper code.

  Returns:
    A decorated function.
  """

  def decorator(arg):
    if category is None:
      arg.__deprecated__ = msg
      return arg
    elif isinstance(arg, type):
      original_new = arg.__new__
      has_init = arg.__init__ is not object.__init__

      @functools.wraps(original_new)
      def wrapped_new(cls, *args, **kwargs):
        warnings.warn(msg, category=category, stacklevel=stacklevel + 1)
        # Mirrors a similar check in object.__new__.
        if not has_init and (args or kwargs):
          raise TypeError(f'{cls.__name__}() takes no arguments')
        if original_new is not object.__new__:
          return original_new(cls, *args, **kwargs)
        else:
          return original_new(cls)

      arg.__new__ = staticmethod(wrapped_new)
      arg.__deprecated__ = wrapped_new.__deprecated__ = msg
      return arg
    elif callable(arg):

      @functools.wraps(arg)
      def wrapper(*args, **kwargs):
        warnings.warn(msg, category=category, stacklevel=stacklevel + 1)
        return arg(*args, **kwargs)

      arg.__deprecated__ = wrapper.__deprecated__ = msg
      return wrapper
    else:
      raise TypeError(
          '@deprecated decorator with non-None category must be applied to '
          f'a class or callable, not {arg!r}'
      )

  return decorator
