# Copyright 2021, The TensorFlow Federated Authors.
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
"""Utilities for working with file systems."""

import asyncio
from collections.abc import Callable
import functools
import os
import random
from typing import Any, Union

import tensorflow as tf
import tree

from tensorflow_federated.python.common_libs import py_typecheck


class FileAlreadyExistsError(Exception):
  pass


def _create_async_def(fn: Callable[..., Any]) -> Callable[..., Any]:
  """A decorator for creating async defs from synchronous functions."""

  @functools.wraps(fn)
  async def wrapper(*args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args, **kwargs)

  return wrapper


exists = _create_async_def(tf.io.gfile.exists)
listdir = _create_async_def(tf.io.gfile.listdir)
makedirs = _create_async_def(tf.io.gfile.makedirs)
rmtree = _create_async_def(tf.io.gfile.rmtree)


def _wrap_as_variable(value):
  """Wraps a value as a `tf.Variable`, if possible."""
  try:
    variable = tf.Variable(initial_value=value)
    return variable
  except ValueError:  # Raised if x is not compatible with `tf.Variable`
    return value


def _read_value(value):
  if isinstance(value, tf.Variable):
    return value.read_value()
  else:
    return value


class _ValueModule(tf.Module):
  """A `tf.Module` wrapping a structure."""

  def __init__(self, value: Any):
    super().__init__()
    # We push leaf values to `tf.Variable` if possible so that they are
    # serialized separately, instead of as a single large proto.
    self._values = tree.map_structure(_wrap_as_variable, value)

  @tf.function(input_signature=())
  def __call__(self) -> Any:
    return tree.map_structure(_read_value, self._values)


async def read_saved_model(path: Union[str, os.PathLike[str]]) -> Any:
  """Reads a SavedModel from `path`."""
  py_typecheck.check_type(path, (str, os.PathLike))

  def _read_saved_model(path: Union[str, os.PathLike[str]]) -> Any:
    if isinstance(path, os.PathLike):
      path = os.fspath(path)
    module = tf.saved_model.load(path)
    return module()

  loop = asyncio.get_running_loop()
  return await loop.run_in_executor(None, _read_saved_model, path)


async def write_saved_model(
    value: Any, path: Union[str, os.PathLike[str]], overwrite: bool = False
) -> None:
  """Writes `value` to `path` using the SavedModel format."""
  py_typecheck.check_type(path, (str, os.PathLike))
  py_typecheck.check_type(overwrite, bool)

  def _write_saved_model(
      value: Any, path: Union[str, os.PathLike[str]], overwrite: bool = False
  ) -> None:
    if isinstance(path, os.PathLike):
      path = os.fspath(path)

    # Create a temporary directory.
    temp_path = f'{path}_temp{random.randint(1000, 9999)}'
    if tf.io.gfile.exists(temp_path):
      tf.io.gfile.rmtree(temp_path)
    tf.io.gfile.makedirs(temp_path)

    # Write to the temporary directory.
    module = _ValueModule(value)
    tf.saved_model.save(module, temp_path, signatures={})

    # Rename the temporary directory to the final location atomically.
    if tf.io.gfile.exists(path):
      if not overwrite:
        raise FileAlreadyExistsError(f'File already exists for path: {path}')
      tf.io.gfile.rmtree(path)
    tf.io.gfile.rename(temp_path, path)

  loop = asyncio.get_running_loop()
  await loop.run_in_executor(None, _write_saved_model, value, path, overwrite)
