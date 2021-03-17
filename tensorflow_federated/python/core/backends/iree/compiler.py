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
"""A collection of utilities for compiling TFF code for execution on IREE."""

import tempfile

import iree.compiler.tf
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.backends.iree import computation_module
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def import_tensorflow_computation(comp, name='fn'):
  """Creates a `computation_module.ComputationModule` from a TF computation.

  WARNING: This helper function is under construction, and most capabilities are
  not implemented at this stage:

  * The parameter and result of `comp` can only be a single tensor. Named
    tuples, sequences, or functional types are not currently supported.

  * Only tensorflow code can be imported.

  TODO(b/153499219): Add support for named tuples, sequences, and functions.

  Args:
    comp: An instance of a `pb.Computation` with TensorFlow code to import.
    name: An optional `str` name of the (single) function in the IREE module.

  Returns:
    An instance of `Module` with the imported function present.

  Raises:
    TypeError: If arguments are of the wrong types, e.g., in `comp` is not a
      TensorFlow computation.
  """
  py_typecheck.check_type(comp, pb.Computation)
  type_spec = type_serialization.deserialize_type(comp.type)
  if not type_spec.is_function():
    type_spec = computation_types.FunctionType(None, type_spec)

  # TODO(b/153499219): Replace this with a recursive check of the signature
  # after relaxing the type restrictions and introducing nested structures.
  py_typecheck.check_type(type_spec.result, computation_types.TensorType)
  if type_spec.parameter is not None:
    py_typecheck.check_type(type_spec.parameter, computation_types.TensorType)

  which_computation = comp.WhichOneof('computation')
  if which_computation != 'tensorflow':
    raise TypeError('Expected a TensorFlow computation, found {}.'.format(
        which_computation))

  output_tensor_names = tensorflow_utils.extract_tensor_names_from_binding(
      comp.tensorflow.result)
  if type_spec.parameter is not None:
    input_tensor_names = tensorflow_utils.extract_tensor_names_from_binding(
        comp.tensorflow.parameter)
  else:
    input_tensor_names = []

  graph_def = serialization_utils.unpack_graph_def(comp.tensorflow.graph_def)
  init_op = comp.tensorflow.initialize_op
  return_elements = input_tensor_names + output_tensor_names
  if init_op:
    graph_def = tensorflow_utils.add_control_deps_for_init_op(
        graph_def, init_op)
    return_elements.append(init_op)

  with tf.Graph().as_default() as graph:
    # TODO(b/153499219): See if we can reintroduce uniquify_shared_names().
    # Right now, it causes loader breakage, and unclear if still necessary.
    import_results = tf.import_graph_def(
        graph_def, input_map={}, return_elements=return_elements, name='')

  if init_op:
    initializer = import_results[-1]
    import_results.pop()
  else:
    initializer = None

  inputs = import_results[0:len(input_tensor_names)]
  outputs = import_results[len(input_tensor_names):]

  with graph.as_default():
    # TODO(b/153499219): Find a way to reflect the nested parameter and result
    # structure here after relaxing the restrictions.
    if inputs:
      assert len(inputs) < 2
      input_dict = {
          'parameter':
              tf.compat.v1.saved_model.utils.build_tensor_info(inputs[0])
      }
    else:
      input_dict = {}
    assert len(outputs) == 1
    output_dict = {
        'result': tf.compat.v1.saved_model.utils.build_tensor_info(outputs[0])
    }
    sig_def = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
        inputs=input_dict, outputs=output_dict, method_name=name)
    with tempfile.TemporaryDirectory() as model_dir:
      builder = tf.compat.v1.saved_model.Builder(model_dir)
      with tf.compat.v1.Session(graph=graph) as sess:
        builder.add_meta_graph_and_variables(
            sess, ['unused'],
            signature_def_map={name: sig_def},
            legacy_init_op=initializer,
            strip_default_attrs=True)
        builder.save()
      iree_module = iree.compiler.tf.compile_saved_model(
          model_dir,
          import_type='SIGNATURE_DEF',
          import_only=True,
          saved_model_tags=set(['unused']),
          exported_names=[name])
      return computation_module.ComputationModule(iree_module, name, type_spec)
