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
"""Writes a lambda GraphDef to a file for testing dataset reduction."""

from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf

flags.DEFINE_string('output', None, 'The path to the output file.')
FLAGS = flags.FLAGS

DATASET_INPUT_TENSOR_NAME = 'input_dataset_placeholder'
RESULT_OUTPUT_TENSOR_NAME = 'result_tensor'


def make_graph():
  """Builds and returns a `tf.Graph` performing dataset reduction."""
  graph = tf.Graph()
  with graph.as_default():
    # Create a placeholder with a fixed name to allow the code running the graph
    # to provide input.
    dataset_input = tf.compat.v1.placeholder(
        name=DATASET_INPUT_TENSOR_NAME, dtype=tf.dtypes.variant)
    dataset = tf.data.experimental.from_variant(
        dataset_input, tf.TensorSpec(shape=(), dtype=tf.int64))
    result = dataset.reduce(
        initial_state=tf.convert_to_tensor(0, tf.int64),
        reduce_func=lambda x, y: x + y,
    )
    # Create a tensor with a fixed name to allow the code running the graph to
    # receive output.
    tf.identity(result, name=RESULT_OUTPUT_TENSOR_NAME)
  return graph


def main(argv: Sequence[str]) -> None:
  del argv
  graph_def_str = str(make_graph().as_graph_def())
  with open(FLAGS.output, 'w') as output_file:
    output_file.write(graph_def_str)


if __name__ == '__main__':
  flags.mark_flag_as_required('output')
  app.run(main)
