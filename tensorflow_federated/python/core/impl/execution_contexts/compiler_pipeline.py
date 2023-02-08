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
"""A pipeline that reduces computations into an executable form."""

from collections.abc import Callable
import functools
from typing import Any, Generic, TypeVar

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.computation import computation_base

_Computation = TypeVar('_Computation', bound=computation_base.Computation)


class CompilerPipeline(Generic[_Computation]):
  """An interface for generating executable artifacts.

  The `CompilerPipeline` holds very little logic; caching for the
  artifacts it generates and essentially nothing else. The `CompilerPipeline`
  is initialized with a `compiler_fn`, to which the pipeline itself delegates
  all the actual work of compilation.

  Different TFF backends may accept different executable artifacts; e.g. a
  backend that supports only a map-reduce execution model may accept instances
  of `tff.backends.mapreduce.MapReduceForm`. The TFF representation of such a
  backend takes the form of an instance of `tff.framework.SyncContext` or
  `tff.framework.AsyncContext`, which would be initialized with a
  `CompilerPipeline` whose `compilation_fn` accepts a `tff.Computation` and
  returns `tff.backends.mapreduce.MapReduceForm`s.
  """

  def __init__(self, compiler_fn: Callable[[_Computation], Any]):
    py_typecheck.check_callable(compiler_fn)
    self._compiler_fn = compiler_fn

  @functools.lru_cache()
  def compile(self, comp: _Computation) -> Any:
    """Compiles `comp`."""
    py_typecheck.check_type(comp, computation_base.Computation)
    return self._compiler_fn(comp)
