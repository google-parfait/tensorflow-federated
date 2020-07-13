# Copyright 2019, The TensorFlow Federated Authors.
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
"""A library of transformation functions for tensorflow computation."""

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def prune_tensorflow_proto(proto):
  """Extracts subgraph from `proto` preserving parameter, result and initialize.

  Args:
    proto: Instance of `pb.Computation` of the `tensorflow` variety whose
      `graphdef` attribute we wish to prune of extraneous ops.

  Returns:
    A transformed instance of `pb.Computation` of the `tensorflow` variety,
    whose `graphdef` attribute contains only ops which can reach the
    parameter or result bindings, or initialize op.
  """
  py_typecheck.check_type(proto, pb.Computation)
  computation_oneof = proto.WhichOneof('computation')
  if computation_oneof != 'tensorflow':
    raise TypeError('`prune_tensorflow_proto` only accepts `Computation` '
                    'protos of the \'tensorflow\' variety; you have passed '
                    'one of variety {}.'.format(computation_oneof))
  if proto.tensorflow.parameter.WhichOneof('binding'):
    parameter_tensor_names = tensorflow_utils.extract_tensor_names_from_binding(
        proto.tensorflow.parameter)
    parameter_names = [
        ':'.join(x.split(':')[:-1]) for x in parameter_tensor_names
    ]
  else:
    parameter_names = []
  return_tensor_names = tensorflow_utils.extract_tensor_names_from_binding(
      proto.tensorflow.result)
  return_names = [':'.join(x.split(':')[:-1]) for x in return_tensor_names]
  graph_def = serialization_utils.unpack_graph_def(proto.tensorflow.graph_def)
  init_op_name = proto.tensorflow.initialize_op
  names_to_preserve = parameter_names + return_names
  if init_op_name:
    names_to_preserve.append(init_op_name)
  subgraph_def = tf.compat.v1.graph_util.extract_sub_graph(
      graph_def, names_to_preserve)
  tf_block = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(subgraph_def),
      initialize_op=proto.tensorflow.initialize_op,
      parameter=proto.tensorflow.parameter,
      result=proto.tensorflow.result)
  pruned_proto = pb.Computation(type=proto.type, tensorflow=tf_block)
  return pruned_proto
