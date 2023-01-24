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
from typing import TypeVar
import warnings

from absl import logging

R = TypeVar('R')


def _add_decorator(fn: Callable[..., R], message: str) -> Callable[..., R]:
  @functools.wraps(fn)
  def wrapper(*args, **kwargs) -> R:
    warnings.warn(message=message, category=DeprecationWarning)
    logging.warning('Deprecation: %s', message)
    return fn(*args, **kwargs)

  return wrapper


def deprecated(*args):
  """Annotates a method as deprecated.

  Supports two different usages:

  1. Decorate an existing function, taking only a single string argument:

     ```
     @deprecated('this method is deprecated')
     def foo():
       ...
     ```

  2. Called directly on an exist function, taking the function first and the
     string message as a second argument:

     ```
     def foo():
       ...

     foo = deprecated(foo, 'this method is deprecated')
     ```

  Args:
    *args: A tuple of one `str` argument that is the message to display when
      calling the decorated method, or a 2-tuple of `callable` and `str`.

  Returns:
    A decorating method for single argument calls, or a wrapped function for
    two argument calls.
  """

  if len(args) == 1:
    message = args[0]
    if not isinstance(message, str):
      raise ValueError(
          'When using `deprecated` as a decorator, the first '
          f'argument must be a `str`, got a: {type(message)}.'
      )
    return functools.partial(_add_decorator, message=message)
  elif len(args) == 2:
    fn, message = args
    return _add_decorator(fn, message)
  else:
    raise ValueError(
        '`deprecated` only takes one or two positional arguments. '
        f'Got arguments: {args}'
    )
