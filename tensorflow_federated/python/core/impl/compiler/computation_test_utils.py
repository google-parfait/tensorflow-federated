# Copyright 2022, The TensorFlow Federated Authors.
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
"""Utilities for testing computations."""

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def create_computation_tensorflow_add_one() -> pb.Computation:
  """Returns a TensorFlow `pb.Computation` adding one to a value."""
  type_spec = tf.int32

  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', type_spec, graph)
    result_value = tf.add(parameter_value, 1)

  result_type, result_binding = tensorflow_utils.capture_result_from_graph(
      result_value, graph)
  type_signature = computation_types.FunctionType(type_spec, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)


def create_computation_tensorflow_add_values(
    type_spec: tf.dtypes.DType = tf.int32) -> pb.Computation:
  """Returns a TensorFlow `pb.Computation` adding two values."""

  with tf.Graph().as_default() as graph:
    parameter_1_value, parameter_1_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', type_spec, graph)
    parameter_2_value, parameter_2_binding = tensorflow_utils.stamp_parameter_in_graph(
        'y', type_spec, graph)
    result_value = tf.add(parameter_1_value, parameter_2_value)

  result_type, result_binding = tensorflow_utils.capture_result_from_graph(
      result_value, graph)
  type_signature = computation_types.FunctionType([type_spec, type_spec],
                                                  result_type)
  struct_binding = pb.TensorFlow.StructBinding(
      element=[parameter_1_binding, parameter_2_binding])
  parameter_binding = pb.TensorFlow.Binding(struct=struct_binding)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  return pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)
