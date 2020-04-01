# Lint as: python3
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

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test

tf.compat.v1.enable_v2_behavior()


def construct_example_training_comp():
  """Constructs a `tff.utils.IterativeProcess` via the FL API."""
  np.random.seed(0)

  input_spec = collections.OrderedDict(
      x=tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
      y=tf.TensorSpec(shape=[None, 1], dtype=tf.int32))

  def model_fn():
    """Constructs keras model."""
    keras_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            1,
            activation=tf.nn.softmax,
            kernel_initializer='zeros',
            input_shape=(2,))
    ])

    return tff.learning.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return tff.learning.build_federated_averaging_process(
      model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01))


class CanonicalFormTest(test.TestCase):

  def test_canonical_form_with_learning_structure_contains_only_one_broadcast_and_one_aggregate(
      self):
    ip = construct_example_training_comp()

    cf = tff.backends.mapreduce.get_canonical_form_for_iterative_process(ip)

    # This type spec test actually carries the meaning that TFF's vanilla path
    # to canonical form will broadcast and aggregate exactly one copy of the
    # parameters. So the type test below in fact functions as a regression test
    # for the TFF compiler pipeline.
    # pyformat: disable
    self.assertEqual(
        cf.work.type_signature.formatted_representation(),
        '(<\n'
        '  <\n'
        '    x=float32[?,2],\n'
        '    y=int32[?,1]\n'
        '  >*,\n'
        '  <\n'
        '    <\n'
        '      trainable=<\n'
        '        float32[2,1],\n'
        '        float32[1]\n'
        '      >,\n'
        '      non_trainable=<>\n'
        '    >\n'
        '  >\n'
        '> -> <\n'
        '  <\n'
        '    <\n'
        '      <\n'
        '        <\n'
        '          float32[2,1],\n'
        '          float32[1]\n'
        '        >,\n'
        '        float32\n'
        '      >,\n'
        '      <\n'
        '        float32,\n'
        '        float32\n'
        '      >,\n'
        '      <\n'
        '        float32,\n'
        '        float32\n'
        '      >\n'
        '    >,\n'
        '    <>\n'
        '  >,\n'
        '  <>\n'
        '>)'
    )
    # pyformat: enable

  def test_canonical_form_with_learning_structure_does_not_change_execution_of_iterative_process(
      self):
    ip_1 = construct_example_training_comp()
    cf = tff.backends.mapreduce.get_canonical_form_for_iterative_process(ip_1)
    ip_2 = tff.backends.mapreduce.get_iterative_process_for_canonical_form(cf)

    self.assertEqual(ip_1.initialize.type_signature,
                     ip_2.initialize.type_signature)
    # The next functions type_signatures may not be equal, since we may have
    # appended an empty tuple as client side-channel outputs if none existed.
    self.assertEqual(ip_1.next.type_signature.parameter,
                     ip_2.next.type_signature.parameter)
    self.assertEqual(ip_1.next.type_signature.result[0],
                     ip_2.next.type_signature.result[0])
    self.assertEqual(ip_1.next.type_signature.result[1],
                     ip_2.next.type_signature.result[1])

    sample_batch = collections.OrderedDict(
        x=np.array([[1., 1.]], dtype=np.float32),
        y=np.array([[0]], dtype=np.int32),
    )
    client_data = [sample_batch]
    state_1 = ip_1.initialize()
    server_state_1, server_output_1 = ip_1.next(state_1, [client_data])
    server_state_1_names = anonymous_tuple.name_list(server_state_1)
    server_state_1_arrays = anonymous_tuple.flatten(server_state_1)
    server_output_1_names = [
        x[0] for x in anonymous_tuple.iter_elements(server_output_1)
    ]
    server_output_1_arrays = anonymous_tuple.flatten(server_output_1)
    state_2 = ip_2.initialize()
    server_state_2, server_output_2, _ = ip_2.next(state_2, [client_data])
    server_state_2_names = anonymous_tuple.name_list(server_state_2)
    server_state_2_arrays = anonymous_tuple.flatten(server_state_2)
    server_output_2_names = [
        x[0] for x in anonymous_tuple.iter_elements(server_output_2)
    ]
    server_output_2_arrays = anonymous_tuple.flatten(server_output_2)

    self.assertEmpty(server_state_1.delta_aggregate_state)
    self.assertEmpty(server_state_1.model_broadcast_state)
    self.assertAllEqual(server_state_1_names, server_state_2_names)
    self.assertAllEqual(server_output_1_names, server_output_2_names)
    self.assertAllClose(server_state_1_arrays, server_state_2_arrays)
    self.assertAllClose(server_output_1_arrays[:2], server_output_2_arrays[:2])


if __name__ == '__main__':
  reference_executor = tff.framework.ReferenceExecutor()
  tff.framework.set_default_executor(reference_executor)
  test.main()
