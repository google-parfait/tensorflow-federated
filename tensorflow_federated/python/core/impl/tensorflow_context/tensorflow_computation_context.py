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
"""The implementation of a context to use in building TF computations."""

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.impl import tensorflow_deserialization
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions


class TensorFlowComputationContext(context_base.Context):
  """The context for building TensorFlow computations."""

  def __init__(self, graph):
    py_typecheck.check_type(graph, tf.Graph)
    self._graph = graph
    self._init_ops = []

  @property
  def init_ops(self):
    """Returns a list of init ops for computations invoked in this context."""
    return list(self._init_ops)

  def ingest(self, val, type_spec):
    type_analysis.check_type(val, type_spec)
    return val

  def invoke(self, comp, arg):
    # We are invoking a tff.tf_computation inside of another
    # tf_computation.
    py_typecheck.check_type(comp, computation_base.Computation)
    computation_proto = computation_impl.ComputationImpl.get_proto(comp)
    init_op, result = (
        tensorflow_deserialization.deserialize_and_call_tf_computation(
            computation_proto, arg, self._graph))
    if init_op:
      self._init_ops.append(init_op)
    return type_conversions.type_to_py_container(result,
                                                 comp.type_signature.result)
