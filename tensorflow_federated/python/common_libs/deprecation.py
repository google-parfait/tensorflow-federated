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

import functools
from typing import Callable, TypeVar
import warnings

from absl import logging

R = TypeVar('R')


# Use typing.ParamSpec for Callable arguments after Python 3.10.
def deprecated(fn: Callable[..., R], message: str) -> Callable[..., R]:

  @functools.wraps(fn)
  def wrapper(*args, **kwargs) -> R:
    warnings.warn(message=message, category=DeprecationWarning)
    logging.warning('Deperecation: %s', message)
    return fn(*args, **kwargs)

  return wrapper
