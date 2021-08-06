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

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.mapreduce import forms
from tensorflow_federated.python.core.backends.mapreduce import test_utils
from tensorflow_federated.python.core.impl.types import computation_types


def _test_broadcast_form_computations():
  server_data_type = (tf.int32, tf.int32)
  context_type = tf.int32
  client_data_type = computation_types.SequenceType(tf.float32)

  @computations.tf_computation(server_data_type)
  def compute_server_context(server_data):
    context_for_clients = server_data[0]
    return context_for_clients

  @computations.tf_computation(context_type, client_data_type)
  def client_processing(context, client_data):
    del context
    del client_data
    return 'some string output on the clients'

  return (compute_server_context, client_processing)


def _test_map_reduce_form_computations():

  @computations.tf_computation
  def initialize():
    return tf.constant(0)

  @computations.tf_computation(tf.int32)
  def prepare(server_state):
    del server_state  # Unused
    return tf.constant(1.0)

  @computations.tf_computation(
      computation_types.SequenceType(tf.float32), tf.float32)
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return True, [], [], []

  @computations.tf_computation
  def zero():
    return tf.constant(0), tf.constant(0)

  @computations.tf_computation((tf.int32, tf.int32), tf.bool)
  def accumulate(accumulator, client_update):
    del accumulator  # Unused
    del client_update  # Unused
    return tf.constant(1), tf.constant(1)

  @computations.tf_computation((tf.int32, tf.int32), (tf.int32, tf.int32))
  def merge(accumulator1, accumulator2):
    del accumulator1  # Unused
    del accumulator2  # Unused
    return tf.constant(1), tf.constant(1)

  @computations.tf_computation(tf.int32, tf.int32)
  def report(accumulator):
    del accumulator  # Unused
    return tf.constant(1.0)

  unit_comp = computations.tf_computation(lambda: [])
  bitwidth = unit_comp
  max_input = unit_comp
  modulus = unit_comp
  unit_type = computation_types.to_type([])

  @computations.tf_computation(tf.int32,
                               (tf.float32, unit_type, unit_type, unit_type))
  def update(server_state, global_update):
    del server_state  # Unused
    del global_update  # Unused
    return tf.constant(1), []

  return (initialize, prepare, work, zero, accumulate, merge, report, bitwidth,
          max_input, modulus, update)


def _build_test_map_reduce_form_with_computations(initialize=None,
                                                  prepare=None,
                                                  work=None,
                                                  zero=None,
                                                  accumulate=None,
                                                  merge=None,
                                                  report=None,
                                                  bitwidth=None,
                                                  max_input=None,
                                                  modulus=None,
                                                  update=None):
  (test_initialize, test_prepare, test_work, test_zero, test_accumulate,
   test_merge, test_report, test_bitwidth, test_max_input, test_modulus,
   test_update) = _test_map_reduce_form_computations()
  return forms.MapReduceForm(
      initialize if initialize else test_initialize,
      prepare if prepare else test_prepare,
      work if work else test_work,
      zero if zero else test_zero,
      accumulate if accumulate else test_accumulate,
      merge if merge else test_merge,
      report if report else test_report,
      bitwidth if bitwidth else test_bitwidth,
      max_input if max_input else test_max_input,
      modulus if modulus else test_modulus,
      update if update else test_update,
  )


class BroadcastFormTest(absltest.TestCase):

  def test_init_does_not_raise_type_error(self):
    (compute_server_context,
     client_processing) = _test_broadcast_form_computations()
    try:
      forms.BroadcastForm(compute_server_context, client_processing)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_raises_type_error_with_mismatched_context_type(self):

    @computations.tf_computation(tf.int32)
    def compute_server_context(x):
      return x

    # Note: `tf.float32` here is mismatched with the context type `tf.int32`
    # returned above.
    @computations.tf_computation(tf.float32, tf.int32)
    def client_processing(context, client_data):
      del context
      del client_data
      return 'some string output on the clients'

    with self.assertRaises(TypeError):
      forms.BroadcastForm(compute_server_context, client_processing)


class MapReduceFormTest(absltest.TestCase):

  def test_init_does_not_raise_type_error(self):
    try:
      _build_test_map_reduce_form_with_computations()
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_init_does_not_raise_type_error_with_unknown_dimensions(self):
    server_state_type = computation_types.TensorType(
        shape=[None], dtype=tf.int32)

    @computations.tf_computation
    def initialize():
      # Return a value of a type assignable to, but not equal to
      # `server_state_type`
      return tf.constant([1, 2, 3])

    @computations.tf_computation(server_state_type)
    def prepare(server_state):
      del server_state  # Unused
      return tf.constant(1.0)

    @computations.tf_computation(
        computation_types.SequenceType(tf.float32), tf.float32)
    def work(client_data, client_input):
      del client_data  # Unused
      del client_input  # Unused
      return True, [], [], []

    @computations.tf_computation
    def zero():
      return tf.constant([], dtype=tf.string)

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string), tf.bool)
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return tf.constant(['abc'])

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string),
        computation_types.TensorType(shape=[None], dtype=tf.string))
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return tf.constant(['abc'])

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string))
    def report(accumulator):
      del accumulator  # Unused
      return tf.constant(1.0)

    unit_comp = computations.tf_computation(lambda: [])
    bitwidth = unit_comp
    max_input = unit_comp
    modulus = unit_comp
    unit_type = computation_types.to_type([])

    @computations.tf_computation(server_state_type,
                                 (tf.float32, unit_type, unit_type, unit_type))
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      # Return a new server state value whose type is assignable but not equal
      # to `server_state_type`, and which is different from the type returned
      # by `initialize`.
      return tf.constant([1]), []

    try:
      forms.MapReduceForm(initialize, prepare, work, zero, accumulate, merge,
                          report, bitwidth, max_input, modulus, update)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_init_raises_type_error_with_bad_initialize_result_type(self):

    @computations.tf_computation
    def initialize():
      return tf.constant(0.0)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(initialize=initialize)

  def test_init_raises_type_error_with_bad_prepare_parameter_type(self):

    @computations.tf_computation(tf.float32)
    def prepare(server_state):
      del server_state  # Unused
      return tf.constant(1.0)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(prepare=prepare)

  def test_init_raises_type_error_with_bad_prepare_result_type(self):

    @computations.tf_computation(tf.int32)
    def prepare(server_state):
      del server_state  # Unused
      return tf.constant(1)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(prepare=prepare)

  def test_init_raises_type_error_with_bad_work_second_parameter_type(self):

    @computations.tf_computation(
        computation_types.SequenceType(tf.float32), tf.int32)
    def work(client_data, client_input):
      del client_data  # Unused
      del client_input  # Unused
      return True, []

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(work=work)

  def test_init_raises_type_error_with_bad_work_result_type(self):

    @computations.tf_computation(
        computation_types.SequenceType(tf.float32), tf.float32)
    def work(client_data, client_input):
      del client_data  # Unused
      del client_input  # Unused
      return tf.constant('abc'), []

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(work=work)

  def test_init_raises_type_error_with_bad_zero_result_type(self):

    @computations.tf_computation
    def zero():
      return tf.constant(0.0), tf.constant(0)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(zero=zero)

  def test_init_raises_type_error_with_bad_accumulate_first_parameter_type(
      self):

    @computations.tf_computation((tf.float32, tf.int32), tf.bool)
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return tf.constant(1), tf.constant(1)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(accumulate=accumulate)

  def test_init_raises_type_error_with_bad_accumulate_second_parameter_type(
      self):

    @computations.tf_computation((tf.float32, tf.float32), tf.string)
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return tf.constant(1), tf.constant(1)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(accumulate=accumulate)

  def test_init_raises_type_error_with_bad_accumulate_result_type(self):

    @computations.tf_computation((tf.float32, tf.float32), tf.bool)
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return tf.constant(1.0), tf.constant(1)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(accumulate=accumulate)

  def test_init_raises_type_error_with_bad_merge_first_parameter_type(self):

    @computations.tf_computation((tf.float32, tf.int32), (tf.int32, tf.int32))
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return tf.constant(1), tf.constant(1)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(merge=merge)

  def test_init_raises_type_error_with_bad_merge_second_parameter_type(self):

    @computations.tf_computation((tf.int32, tf.int32), (tf.float32, tf.int32))
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return tf.constant(1), tf.constant(1)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(merge=merge)

  def test_init_raises_type_error_with_bad_merge_result_type(self):

    @computations.tf_computation((tf.int32, tf.int32), (tf.int32, tf.int32))
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return tf.constant(1.0), tf.constant(1)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(merge=merge)

  def test_init_raises_type_error_with_bad_report_parameter_type(self):

    @computations.tf_computation(tf.float32, tf.int32)
    def report(accumulator):
      del accumulator  # Unused
      return tf.constant(1.0)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(report=report)

  def test_init_raises_type_error_with_bad_report_result_type(self):

    @computations.tf_computation(tf.int32, tf.int32)
    def report(accumulator):
      del accumulator  # Unused
      return tf.constant(1)

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(report=report)

  def test_init_raises_type_error_with_bad_update_first_parameter_type(self):

    @computations.tf_computation(tf.float32,
                                 (tf.float32, computation_types.StructType([])))
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      return tf.constant(1), []

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(update=update)

  def test_init_raises_type_error_with_bad_update_second_parameter_type(self):

    @computations.tf_computation(tf.int32,
                                 (tf.int32, computation_types.StructType([])))
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      return tf.constant(1), []

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(update=update)

  def test_init_raises_type_error_with_bad_update_result_type(self):

    @computations.tf_computation(tf.int32,
                                 (tf.float32, computation_types.StructType([])))
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      return tf.constant(1.0), []

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(update=update)

  def test_securely_aggregates_tensors_true(self):
    cf_with_secure_sum = test_utils.get_federated_sum_example(secure_sum=True)
    self.assertTrue(cf_with_secure_sum.securely_aggregates_tensors)

  def test_securely_aggregates_tensors_false(self):
    cf_with_no_secure_sum = test_utils.get_federated_sum_example(
        secure_sum=False)
    self.assertFalse(cf_with_no_secure_sum.securely_aggregates_tensors)

  def test_summary(self):
    mrf = test_utils.get_temperature_sensor_example()

    class CapturePrint(object):

      def __init__(self):
        self.summary = ''

      def __call__(self, msg):
        self.summary += msg + '\n'

    capture = CapturePrint()
    mrf.summary(print_fn=capture)
    # pyformat: disable
    self.assertEqual(
        capture.summary,
        'initialize: ( -> <num_rounds=int32>)\n'
        'prepare   : (<num_rounds=int32> -> <max_temperature=float32>)\n'
        'work      : (<data=float32*,state=<max_temperature=float32>> -> <<is_over=bool>,<>,<>,<>>)\n'
        'zero      : ( -> <num_total=int32,num_over=int32>)\n'
        'accumulate: (<accumulator=<num_total=int32,num_over=int32>,update=<is_over=bool>> -> <num_total=int32,num_over=int32>)\n'
        'merge     : (<accumulator1=<num_total=int32,num_over=int32>,accumulator2=<num_total=int32,num_over=int32>> -> <num_total=int32,num_over=int32>)\n'
        'report    : (<num_total=int32,num_over=int32> -> <ratio_over_threshold=float32>)\n'
        'secure_sum_bitwidth: ( -> <>)\n'
        'secure_sum_max_input: ( -> <>)\n'
        'secure_modular_sum_modulus: ( -> <>)\n'
        'update    : (<state=<num_rounds=int32>,update=<<ratio_over_threshold=float32>,<>,<>,<>>> -> <<num_rounds=int32>,<ratio_over_threshold=float32>>)\n'
    )
    # pyformat: enable


if __name__ == '__main__':
  absltest.main()
