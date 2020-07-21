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
import functools

from typing import Callable, Any

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base


class CompilerPipeline(object):
  """An interface for generating executable artifacts.

  The `CompilerPipeline` holds very little logic; caching for the
  artifacts it generates and essentially nothing else. The `CompilerPipeline`
  is initialized with a `compilation_function`, to which the pipeline itself
  delegates all the actual work of compilation.

  Different TFF backends may accept different executable artifacts; e.g. a
  backend that supports only a map-reduce execution model may accept instances
  of `tff.backends.mapreduce.CanonicalForm`. The TFF representation of such a
  backend takes the form of an instance of `tff.framework.Context`, which would
  be initialized with a `CompilerPipeline` whose `compilation_fn` accepts
  `tff.Computations` and returns CanonicalForms.
  """

  def __init__(self, compilation_fn: Callable[[computation_base.Computation],
                                              Any]):
    py_typecheck.check_callable(compilation_fn)
    self._compilation_fn = compilation_fn

  @functools.lru_cache()
  def compile(self, computation_to_compile: computation_base.Computation):
    """Generates executable for `computation_to_compile`."""
    py_typecheck.check_type(computation_to_compile,
                            computation_base.Computation)
    return self._compilation_fn(computation_to_compile)
