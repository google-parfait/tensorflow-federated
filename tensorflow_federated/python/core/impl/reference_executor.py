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
"""A simple interpreted reference executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import executor_base


class ReferenceExecutor(executor_base.Executor):
  """A simple interpreted reference executor.

  This executor is to be used by default in unit tests and simple applications
  such as colab notebooks and turorials. It is intended to serve as the gold
  standard of correctness for all other executors to compare against. As such,
  it is designed for simplicity and ease of reasoning about correctness, rather
  than for high performance. We will tolerate copying values, marshaling and
  unmarshaling when crossing TF graph boundary, etc., for the sake of keeping
  the logic minimal. The executor can be reused across multiple calls, so any
  state associated with individual executions is maintained separately from
  this class. High-performance simulations on large data sets will require a
  separate executor optimized for performance. This executor is plugged in as
  the handler of computation invocations at the top level of the context stack.
  """

  def __init__(self):
    """Creates a reference executor."""

    # TODO(b/113116813): Add a way to declare environmental bindings here,
    # e.g., a way to specify how data URIs are mapped to physical resources.
    pass

  def execute(self, computation_proto):
    """Runs the given self-contained computation, and returns the final results.

    Args:
      computation_proto: An instance of `pb.Computation()` to execute. It must
        be self-contained, i.e., it must not declare any parameters (all inputs
        it needs should be baked into its body by the time it is submitted for
        execution by the default execution context that spawns it).

    Returns:
      The result produced by the computation (the format to be described).

    Raises:
      NotImplementedError: At the moment, every time when invoked.
    """
    py_typecheck.check_type(computation_proto, pb.Computation)
    raise NotImplementedError('The reference executor is not implemented yet.')
