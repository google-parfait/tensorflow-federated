# Copyright 2020, The TensorFlow Federated Authors.
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
"""TFF-independent data class for the data TFF keeps in its TF protos."""
from collections.abc import Sequence
from typing import Optional

import attr
import tensorflow as tf


def _check_names_are_strings(instance, attribute, value):
  del instance  # Unused.
  for name in value:
    if not isinstance(name, str):
      raise TypeError(
          'Each entry in {} must be of string type; '
          'encountered an element of type {}'.format(attribute, type(name))
      )


@attr.s(frozen=True, eq=False)
class GraphSpec:
  """Container class for validating input to graph merging functions.

  Mirrors the serialization format of TFF, the main difference being that
  `GraphSpec` takes an already flattened list of names as `in_names` and
  `out_names`, as opposed to the single binding taken by TFF serialization.

  Attributes:
    graph_def: Instance of `tf.compat.v1.GraphDef`.
    init_op: Either a string name of the init op in the graph represented by
      `graph_def`, or `None` if there is no such init op.
    in_names: A Python `list` or `tuple of string names corresponding to the
      input objects in `graph_def`. Must be empty if there are no such inputs.
    out_names: An iterable of string names of the output tensors in `graph_def`,
      subject to the same restrictions as `in_names`.
  """

  graph_def: tf.compat.v1.GraphDef = attr.ib(
      validator=attr.validators.instance_of(tf.compat.v1.GraphDef)
  )
  init_op: Optional[str] = attr.ib(
      validator=attr.validators.instance_of((str, type(None)))
  )
  in_names: Sequence[str] = attr.ib(validator=_check_names_are_strings)
  out_names: Sequence[str] = attr.ib(validator=_check_names_are_strings)

  def to_meta_graph_def(self):
    """Packs `GraphSpec` into a `tf.compat.v1.MetaGraphDef`.

    Does not adjust names in the graph_def which backs this instance of
    `GraphSpec`. Only introduces one important convention: the initialize op
    is stored in the graph collection with key `tf.compat.v1.GraphKeys.INIT_OP`.

    Returns:
      A `tf.compat.v1.MetaGraphDef` which represents the logic in this
      `GraphSpec`, with the convention that the initialize op is stored in the
      graph collection with key `tf.compat.v1.GraphKeys.INIT_OP`, and the input
      and output tensor information is stored in the lone populated
      `SignatureDef`.
    """

    with tf.Graph().as_default() as graph_for_tensor_specs:
      tf.graph_util.import_graph_def(self.graph_def, name='')

    def _get_tensor_spec(name):
      return tf.TensorSpec.from_tensor(
          graph_for_tensor_specs.get_tensor_by_name(name)
      )

    in_names_to_tensor_specs = {
        name: _get_tensor_spec(name) for name in self.in_names
    }
    out_names_to_tensor_specs = {
        name: _get_tensor_spec(name) for name in self.out_names
    }

    meta_graph_def = tf.compat.v1.MetaGraphDef()

    meta_graph_def.graph_def.CopyFrom(self.graph_def)

    if self.init_op is not None:
      meta_graph_def.collection_def[
          tf.compat.v1.GraphKeys.INIT_OP
      ].node_list.value.append(self.init_op)

    signature_def = meta_graph_def.signature_def['FunctionSpec']
    for index, input_name in enumerate(self.in_names):
      input_tensor_info = signature_def.inputs['arg_{}'.format(index)]
      input_tensor_info.name = input_name
      input_tensor_info.dtype = in_names_to_tensor_specs[
          input_name
      ].dtype.as_datatype_enum
      input_tensor_info.tensor_shape.CopyFrom(
          in_names_to_tensor_specs[input_name].shape.as_proto()
      )
    for index, output_name in enumerate(self.out_names):
      output_tensor_info = signature_def.outputs['output_{}'.format(index)]
      output_tensor_info.name = output_name
      output_tensor_info.dtype = out_names_to_tensor_specs[
          output_name
      ].dtype.as_datatype_enum
      output_tensor_info.tensor_shape.CopyFrom(
          out_names_to_tensor_specs[output_name].shape.as_proto()
      )
    return meta_graph_def
