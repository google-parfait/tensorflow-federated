# Copyright 2021 The TensorFlow Federated Authors
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
"""Writes out a GraphDef which performs a reduce over a structural dataset."""

import operator
from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf

flags.DEFINE_string('output', None, 'The path to the output file.')
FLAGS = flags.FLAGS

DATASET_INPUT_TENSOR_NAME = 'serialized_dataset_input'
OUTPUT_TENSOR_NAMES = ['output_tensor_0', 'output_tensor_1', 'output_tensor_2']


def make_graph():
  """Builds and returns a `tf.Graph` performing dataset reduction."""
  graph = tf.Graph()
  with graph.as_default():
    # Create a placeholder with a fixed name to allow the code running the graph
    # to provide input.
    serialized_dataset_input = tf.compat.v1.placeholder(
        name=DATASET_INPUT_TENSOR_NAME, dtype=tf.string)
    variant_dataset = tf.raw_ops.DatasetFromGraph(
        graph_def=serialized_dataset_input)
    int_spec = tf.TensorSpec(shape=(), dtype=tf.int64)
    # Structure the inputs using dictionaries and lists to check that layout
    # of structural inputs works properly.
    dataset = tf.data.experimental.from_variant(variant_dataset, {
        'a': int_spec,
        'nested': (int_spec, int_spec)
    })
    zero = tf.convert_to_tensor(0, tf.int64)

    def sum_structure(first, second):
      return tf.nest.map_structure(operator.add, first, second)

    result = dataset.reduce(
        initial_state={
            'a': zero,
            'nested': (zero, zero)
        },
        reduce_func=sum_structure)
    # Create tensors with fixed names to allow the code running the graph to
    # receive output.
    tf.identity(result['a'], name=OUTPUT_TENSOR_NAMES[0])
    tf.identity(result['nested'][0], name=OUTPUT_TENSOR_NAMES[1])
    tf.identity(result['nested'][1], name=OUTPUT_TENSOR_NAMES[2])
  return graph


def main(argv: Sequence[str]) -> None:
  del argv
  graph_def_str = str(make_graph().as_graph_def())
  with open(FLAGS.output, 'w') as output_file:
    output_file.write(graph_def_str)


if __name__ == '__main__':
  flags.mark_flag_as_required('output')
  app.run(main)
