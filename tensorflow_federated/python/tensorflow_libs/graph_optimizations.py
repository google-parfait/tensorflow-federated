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
"""Library of static graph optimizations."""

from absl import logging

from tensorflow.python.grappler import tf_optimizer
from tensorflow_federated.python.tensorflow_libs import graph_spec


def optimize_graph_spec(graph_spec_obj, config_proto):
  """Applies Grappler with given options to a `graph_spec.GraphSpec`.

  For more information on Grappler, see
  https://www.tensorflow.org/guide/graph_optimization

  Args:
    graph_spec_obj: Instance of `graph_spec.GraphSpec` representing the
      TensorFlow computation to optimize.
    config_proto: Instance of `tf.compat.v1.ConfigProto` specifying optimization
      options for Grappler.

  Returns:
    An instance of `graph_spec_obj` which has been passed through Grappler and
    optimized if possible.
  """
  meta_graph_def = graph_spec_obj.to_meta_graph_def()

  try:
    # Grappler raises if it fails to find feeds and fetches, but can handle
    # *some* no-arg graphs, so we try/except here.
    optimized_graph_def = tf_optimizer.OptimizeGraph(
        config_proto, meta_graph_def
    )
  except ValueError as error:
    logging.info(
        (
            'Grappler has raised the error %s; falling back to using '
            'non-optimized graph.'
        ),
        error,
    )
    optimized_graph_def = graph_spec_obj.graph_def

  return graph_spec.GraphSpec(
      optimized_graph_def,
      init_op=graph_spec_obj.init_op,
      in_names=graph_spec_obj.in_names,
      out_names=graph_spec_obj.out_names,
  )
