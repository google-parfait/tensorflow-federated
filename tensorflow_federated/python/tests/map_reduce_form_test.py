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


def construct_example_training_comp() -> tff.learning.templates.LearningProcess:
  """Constructs a `tff.learning.templates.LearningProcess` via the FL API."""
  np.random.seed(0)

  input_spec = collections.OrderedDict(
      x=tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
      y=tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
  )

  def model_fn():
    """Constructs keras model."""
    keras_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                1,
                activation=tf.nn.softmax,
                kernel_initializer='zeros',
                input_shape=(2,),
            )
        ]
    )

    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

  return tff.learning.algorithms.build_weighted_fed_avg(
      model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
  )


class MapReduceFormTest(tf.test.TestCase):

  def test_map_reduce_form_with_learning_structure_contains_only_one_broadcast_and_one_aggregate(
      self,
  ):
    ip = construct_example_training_comp()

    cf = tff.backends.mapreduce.get_map_reduce_form_for_computation(ip.next)

    # This type spec test actually carries the meaning that TFF's vanilla path
    # to canonical form will broadcast and aggregate exactly one copy of the
    # parameters. So the type test below in fact functions as a regression test
    # for the TFF compiler pipeline.
    # pyformat: disable
    self.assertEqual(
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
        '      sparse_categorical_accuracy=<\n'
        '        float32,\n'
        '        float32\n'
        '      >,\n'
        '      loss=<\n'
        '        float32,\n'
        '        float32\n'
        '      >,\n'
        '      num_examples=<\n'
        '        int64\n'
        '      >,\n'
        '      num_batches=<\n'
        '        int64\n'
        '      >\n'
        '    >,\n'
        '    <\n'
        '      float32[2,1],\n'
        '      float32[1]\n'
        '    >,\n'
        '    float32\n'
        '  >,\n'
        '  <>,\n'
        '  <>,\n'
        '  <>\n'
        '>)',
        cf.work.type_signature.formatted_representation())
    # pyformat: enable

  def test_map_reduce_form_with_learning_structure_does_not_change_execution_of_iterative_process(
      self,
  ):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest(
          'b/137602785: bring GPU test back after the fix for `wrap_function`'
      )
    ip_1 = construct_example_training_comp()
    # We disable Grappler to prevent a single TF function from being pulled into
    # the eager TF runtime with multiple definitions.
    grappler_config = tf.compat.v1.ConfigProto()
    grappler_config.graph_options.rewrite_options.disable_meta_optimizer = True
    cf = tff.backends.mapreduce.get_map_reduce_form_for_computation(
        ip_1.next, grappler_config=grappler_config
    )
    next_2 = tff.backends.mapreduce.get_computation_for_map_reduce_form(cf)
    ip_2 = tff.templates.IterativeProcess(ip_1.initialize, next_2)

    ip_1.initialize.type_signature.check_equivalent_to(
        ip_2.initialize.type_signature
    )
    # The next functions type_signatures may not be equal, since we may have
    # appended an empty tuple as client side-channel outputs if none existed.
    ip_1.next.type_signature.parameter.check_equivalent_to(
        ip_2.next.type_signature.parameter
    )
    # Map reduce form can strip out python structures, so we check equivalence
    # against StructType.
    ip_1_result_type = ip_1.next.type_signature.result
    ip_2_result_type = ip_2.next.type_signature.result
    tff.types.StructType(ip_1_result_type).check_equivalent_to(ip_2_result_type)

    sample_batch = collections.OrderedDict(
        x=np.array([[1.0, 1.0]], dtype=np.float32),
        y=np.array([[0]], dtype=np.int32),
    )
    client_data = [sample_batch]
    state_1 = ip_1.initialize()
    ip_1_next_result = ip_1.next(state_1, [client_data])
    server_state_1 = ip_1_next_result.state
    server_output_1 = ip_1_next_result.metrics
    # The serialized representation of `ip` loses the Python containers, so we
    # assert that it matches the odict_or_tuple-ified representations.
    server_state_1 = tff.structure.to_odict_or_tuple(
        tff.structure.from_container(server_state_1, recursive=True)
    )
    server_output_1 = tff.structure.to_odict_or_tuple(
        tff.structure.from_container(server_output_1, recursive=True)
    )
    server_state_1_arrays = tf.nest.flatten(server_state_1)
    server_output_1_arrays = tf.nest.flatten(server_output_1)
    state_2 = ip_2.initialize()
    server_state_2, server_output_2 = ip_2.next(state_2, [client_data])
    server_state_2_arrays = tf.nest.flatten(server_state_2)
    server_output_2_arrays = tf.nest.flatten(server_output_2)

    self.assertEqual(server_state_1['distributor'], ())
    # Note that we cannot simply use assertEqual because the values may differ
    # due to floating point issues.
    tf.nest.assert_same_structure(server_state_1, server_state_2)
    tf.nest.assert_same_structure(server_output_1, server_output_2)
    self.assertAllClose(server_state_1_arrays, server_state_2_arrays)
    self.assertAllClose(server_output_1_arrays[:2], server_output_2_arrays[:2])


if __name__ == '__main__':
  tff.backends.test.set_sync_test_cpp_execution_context()
  tf.test.main()
