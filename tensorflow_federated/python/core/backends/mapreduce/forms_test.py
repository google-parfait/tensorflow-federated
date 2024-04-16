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
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.mapreduce import distribute_aggregate_test_utils
from tensorflow_federated.python.core.backends.mapreduce import forms
from tensorflow_federated.python.core.backends.mapreduce import mapreduce_test_utils
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def _test_broadcast_form_computations():
  server_data_type = (np.int32, np.int32)
  context_type = np.int32
  client_data_type = computation_types.SequenceType(np.float32)

  @tensorflow_computation.tf_computation(server_data_type)
  def compute_server_context(server_data):
    context_for_clients = server_data[0]
    return context_for_clients

  @tensorflow_computation.tf_computation(context_type, client_data_type)
  def client_processing(context, client_data):
    del context
    del client_data
    return 'some string output on the clients'

  return (compute_server_context, client_processing)


def _test_map_reduce_form_computations():

  @tensorflow_computation.tf_computation(np.int32)
  def prepare(server_state):
    del server_state  # Unused
    return 1.0

  @tensorflow_computation.tf_computation(
      computation_types.SequenceType(np.float32), np.float32
  )
  def work(client_data, client_input):
    del client_data  # Unused
    del client_input  # Unused
    return True, [], [], []

  @tensorflow_computation.tf_computation
  def zero():
    return 0, 0

  @tensorflow_computation.tf_computation((np.int32, np.int32), np.bool_)
  def accumulate(accumulator, client_update):
    del accumulator  # Unused
    del client_update  # Unused
    return 1, 1

  @tensorflow_computation.tf_computation(
      (np.int32, np.int32), (np.int32, np.int32)
  )
  def merge(accumulator1, accumulator2):
    del accumulator1  # Unused
    del accumulator2  # Unused
    return 1, 1

  @tensorflow_computation.tf_computation(np.int32, np.int32)
  def report(accumulator):
    del accumulator  # Unused
    return 1.0

  unit_comp = tensorflow_computation.tf_computation(lambda: [])
  bitwidth = unit_comp
  max_input = unit_comp
  modulus = unit_comp
  unit_type = computation_types.to_type([])

  @tensorflow_computation.tf_computation(
      np.int32, (np.float32, unit_type, unit_type, unit_type)
  )
  def update(server_state, global_update):
    del server_state  # Unused
    del global_update  # Unused
    return 1, []

  return (
      prepare,
      work,
      zero,
      accumulate,
      merge,
      report,
      bitwidth,
      max_input,
      modulus,
      update,
  )


def _build_test_map_reduce_form_with_computations(
    type_signature=None,
    prepare=None,
    work=None,
    zero=None,
    accumulate=None,
    merge=None,
    report=None,
    bitwidth=None,
    max_input=None,
    modulus=None,
    update=None,
):
  (
      test_prepare,
      test_work,
      test_zero,
      test_accumulate,
      test_merge,
      test_report,
      test_bitwidth,
      test_max_input,
      test_modulus,
      test_update,
  ) = _test_map_reduce_form_computations()
  type_signature = (
      type_signature
      or mapreduce_test_utils.generate_unnamed_type_signature(
          test_update, test_work
      )
  )
  return forms.MapReduceForm(
      type_signature,
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


def _test_distribute_aggregate_form_computations():

  @federated_computation.federated_computation(
      computation_types.FederatedType(np.int32, placements.SERVER)
  )
  def server_prepare(server_state):
    @tensorflow_computation.tf_computation
    def server_prepare_broadcast_tf():
      return 1.0

    @tensorflow_computation.tf_computation
    def server_prepare_state_tf():
      return 32

    return [
        [
            intrinsics.federated_value(
                server_prepare_broadcast_tf(), placements.SERVER
            )
        ]
    ], [server_prepare_state_tf(), server_state]

  @federated_computation.federated_computation(
      [[computation_types.FederatedType(np.float32, placements.SERVER)]]
  )
  def server_to_client_broadcast(context_at_server):
    return [intrinsics.federated_broadcast(context_at_server[0][0])]

  @federated_computation.federated_computation(
      computation_types.FederatedType(
          computation_types.SequenceType(np.float32), placements.CLIENTS
      ),
      [computation_types.FederatedType(np.float32, placements.CLIENTS)],
  )
  def client_work(client_data, context_at_clients):
    @tensorflow_computation.tf_computation
    def client_work_tf():
      return tf.constant([1, 2])

    del client_data  # Unused
    del context_at_clients  # Unused
    return [[intrinsics.federated_value(client_work_tf(), placements.CLIENTS)]]

  @federated_computation.federated_computation(
      [np.int32, computation_types.FederatedType(np.int32, placements.SERVER)],
      [[
          computation_types.FederatedType(
              computation_types.TensorType(np.int32, [2]), placements.CLIENTS
          )
      ]],
  )
  def client_to_server_aggregation(temp_server_state, client_updates):
    del temp_server_state  # Unused.
    return [intrinsics.federated_secure_sum_bitwidth(client_updates[0][0], 100)]

  @federated_computation.federated_computation(
      [np.int32, computation_types.FederatedType(np.int32, placements.SERVER)],
      [
          computation_types.FederatedType(
              computation_types.TensorType(np.int32, [2]), placements.SERVER
          )
      ],
  )
  def server_result(temp_server_state, aggregated_results):
    del aggregated_results  # Unused
    return temp_server_state[1], intrinsics.federated_value(
        [], placements.SERVER
    )

  return (
      server_prepare,
      server_to_client_broadcast,
      client_work,
      client_to_server_aggregation,
      server_result,
  )


def _build_test_distribute_aggregate_form_with_computations(
    type_signature=None,
    server_prepare=None,
    server_to_client_broadcast=None,
    client_work=None,
    client_to_server_aggregation=None,
    server_result=None,
):
  (
      test_server_prepare,
      test_server_to_client_broadcast,
      test_client_work,
      test_client_to_server_aggregation,
      test_server_result,
  ) = _test_distribute_aggregate_form_computations()
  type_signature = (
      type_signature
      or distribute_aggregate_test_utils.generate_unnamed_type_signature(
          test_server_prepare, test_client_work, test_server_result
      )
  )
  return forms.DistributeAggregateForm(
      type_signature,
      server_prepare if server_prepare else test_server_prepare,
      server_to_client_broadcast
      if server_to_client_broadcast
      else test_server_to_client_broadcast,
      client_work if client_work else test_client_work,
      client_to_server_aggregation
      if client_to_server_aggregation
      else test_client_to_server_aggregation,
      server_result if server_result else test_server_result,
  )


class BroadcastFormTest(absltest.TestCase):

  def test_init_does_not_raise_type_error(self):
    (compute_server_context, client_processing) = (
        _test_broadcast_form_computations()
    )
    try:
      forms.BroadcastForm(compute_server_context, client_processing)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_raises_type_error_with_mismatched_context_type(self):

    @tensorflow_computation.tf_computation(np.int32)
    def compute_server_context(x):
      return x

    # Note: `np.float32` here is mismatched with the context type `np.int32`
    # returned above.
    @tensorflow_computation.tf_computation(np.float32, np.int32)
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
    server_state_type = computation_types.TensorType(np.int32, [None])

    @tensorflow_computation.tf_computation(server_state_type)
    def prepare(server_state):
      del server_state  # Unused
      return 1.0

    @tensorflow_computation.tf_computation(
        computation_types.SequenceType(np.float32), np.float32
    )
    def work(client_data, client_input):
      del client_data  # Unused
      del client_input  # Unused
      return True, [], [], []

    @tensorflow_computation.tf_computation
    def zero():
      return tf.constant([], dtype=tf.string)

    @tensorflow_computation.tf_computation(
        computation_types.TensorType(np.str_, [None]), np.bool_
    )
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return tf.constant(['abc'])

    @tensorflow_computation.tf_computation(
        computation_types.TensorType(np.str_, [None]),
        computation_types.TensorType(np.str_, [None]),
    )
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return tf.constant(['abc'])

    @tensorflow_computation.tf_computation(
        computation_types.TensorType(np.str_, [None])
    )
    def report(accumulator):
      del accumulator  # Unused
      return 1.0

    unit_comp = tensorflow_computation.tf_computation(lambda: [])
    bitwidth = unit_comp
    max_input = unit_comp
    modulus = unit_comp
    unit_type = computation_types.to_type([])

    @tensorflow_computation.tf_computation(
        server_state_type, (np.float32, unit_type, unit_type, unit_type)
    )
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      # Return a new server state value whose type is assignable but not equal
      # to `server_state_type`, and which is different from the type returned
      # by the expected initial state.
      return tf.constant([1]), []

    type_signature = mapreduce_test_utils.generate_unnamed_type_signature(
        update, work
    )
    try:
      forms.MapReduceForm(
          type_signature,
          prepare,
          work,
          zero,
          accumulate,
          merge,
          report,
          bitwidth,
          max_input,
          modulus,
          update,
      )
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_init_raises_type_error_with_bad_prepare_parameter_type(self):

    @tensorflow_computation.tf_computation(np.float32)
    def prepare(server_state):
      del server_state  # Unused
      return 1.0

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(prepare=prepare)

  def test_init_raises_type_error_with_bad_prepare_result_type(self):

    @tensorflow_computation.tf_computation(np.int32)
    def prepare(server_state):
      del server_state  # Unused
      return 1

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(prepare=prepare)

  def test_init_raises_type_error_with_bad_work_second_parameter_type(self):
    @tensorflow_computation.tf_computation(
        computation_types.SequenceType(np.float32), np.int32
    )
    def work(client_data, client_input):
      del client_data  # Unused
      del client_input  # Unused
      return True, []

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(work=work)

  def test_init_raises_type_error_with_bad_work_result_type(self):
    @tensorflow_computation.tf_computation(
        computation_types.SequenceType(np.float32), np.float32
    )
    def work(client_data, client_input):
      del client_data  # Unused
      del client_input  # Unused
      return 'abc', []

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(work=work)

  def test_init_raises_type_error_with_bad_zero_result_type(self):
    @tensorflow_computation.tf_computation
    def zero():
      return 0.0, 0

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(zero=zero)

  def test_init_raises_type_error_with_bad_accumulate_first_parameter_type(
      self,
  ):

    @tensorflow_computation.tf_computation((np.float32, np.int32), np.bool_)
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return 1, 1

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(accumulate=accumulate)

  def test_init_raises_type_error_with_bad_accumulate_second_parameter_type(
      self,
  ):

    @tensorflow_computation.tf_computation((np.float32, np.float32), np.str_)
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return 1, 1

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(accumulate=accumulate)

  def test_init_raises_type_error_with_bad_accumulate_result_type(self):

    @tensorflow_computation.tf_computation((np.float32, np.float32), np.bool_)
    def accumulate(accumulator, client_update):
      del accumulator  # Unused
      del client_update  # Unused
      return 1.0, 1

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(accumulate=accumulate)

  def test_init_raises_type_error_with_bad_merge_first_parameter_type(self):
    @tensorflow_computation.tf_computation(
        (np.float32, np.int32), (np.int32, np.int32)
    )
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return 1, 1

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(merge=merge)

  def test_init_raises_type_error_with_bad_merge_second_parameter_type(self):
    @tensorflow_computation.tf_computation(
        (np.int32, np.int32), (np.float32, np.int32)
    )
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return 1, 1

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(merge=merge)

  def test_init_raises_type_error_with_bad_merge_result_type(self):
    @tensorflow_computation.tf_computation(
        (np.int32, np.int32), (np.int32, np.int32)
    )
    def merge(accumulator1, accumulator2):
      del accumulator1  # Unused
      del accumulator2  # Unused
      return 1.0, 1

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(merge=merge)

  def test_init_raises_type_error_with_bad_report_parameter_type(self):

    @tensorflow_computation.tf_computation(np.float32, np.int32)
    def report(accumulator):
      del accumulator  # Unused
      return 1.0

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(report=report)

  def test_init_raises_type_error_with_bad_report_result_type(self):

    @tensorflow_computation.tf_computation(np.int32, np.int32)
    def report(accumulator):
      del accumulator  # Unused
      return 1

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(report=report)

  def test_init_raises_type_error_with_bad_update_first_parameter_type(self):
    @tensorflow_computation.tf_computation(
        np.float32, (np.float32, computation_types.StructType([]))
    )
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      return 1, []

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(update=update)

  def test_init_raises_type_error_with_bad_update_second_parameter_type(self):
    @tensorflow_computation.tf_computation(
        np.int32, (np.int32, computation_types.StructType([]))
    )
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      return 1, []

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(update=update)

  def test_init_raises_type_error_with_bad_update_result_type(self):
    @tensorflow_computation.tf_computation(
        np.int32, (np.float32, computation_types.StructType([]))
    )
    def update(server_state, global_update):
      del server_state  # Unused
      del global_update  # Unused
      return 1.0, []

    with self.assertRaises(TypeError):
      _build_test_map_reduce_form_with_computations(update=update)

  def test_securely_aggregates_tensors_true(self):
    cf_with_secure_sum = mapreduce_test_utils.get_federated_sum_example(
        secure_sum=True
    ).mrf
    self.assertTrue(cf_with_secure_sum.securely_aggregates_tensors)

  def test_securely_aggregates_tensors_false(self):
    cf_with_no_secure_sum = mapreduce_test_utils.get_federated_sum_example(
        secure_sum=False
    ).mrf
    self.assertFalse(cf_with_no_secure_sum.securely_aggregates_tensors)

  def test_summary(self):
    mrf = mapreduce_test_utils.get_temperature_sensor_example().mrf

    class CapturePrint:

      def __init__(self):
        self.summary = ''

      def __call__(self, msg):
        self.summary += msg + '\n'

    capture = CapturePrint()
    mrf.summary(print_fn=capture)
    # pyformat: disable
    self.assertEqual(
        capture.summary,
        'prepare                   : (<num_rounds=int32> -> <max_temperature=float32>)\n'
        'work                      : (<data=float32*,state=<max_temperature=float32>> -> <<is_over=bool>,<>,<>,<>>)\n'
        'zero                      : ( -> <num_total=int32,num_over=int32>)\n'
        'accumulate                : (<accumulator=<num_total=int32,num_over=int32>,update=<is_over=bool>> -> <num_total=int32,num_over=int32>)\n'
        'merge                     : (<accumulator1=<num_total=int32,num_over=int32>,accumulator2=<num_total=int32,num_over=int32>> -> <num_total=int32,num_over=int32>)\n'
        'report                    : (<num_total=int32,num_over=int32> -> <ratio_over_threshold=float32>)\n'
        'secure_sum_bitwidth       : ( -> <>)\n'
        'secure_sum_max_input      : ( -> <>)\n'
        'secure_modular_sum_modulus: ( -> <>)\n'
        'update                    : (<state=<num_rounds=int32>,update=<<ratio_over_threshold=float32>,<>,<>,<>>> -> <<num_rounds=int32>,<ratio_over_threshold=float32>>)\n'
    )  # pyformat: enable


class DistributeAggregateFormTest(absltest.TestCase):

  def test_init_does_not_raise_type_error(self):
    try:
      daf = _build_test_distribute_aggregate_form_with_computations()
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')
    assert daf.type_signature.is_equivalent_to(
        computation_types.FunctionType(
            computation_types.StructType([
                computation_types.FederatedType(np.int32, placements.SERVER),
                computation_types.FederatedType(
                    computation_types.SequenceType(np.float32),
                    placements.CLIENTS,
                ),
            ]),
            computation_types.StructType([
                computation_types.FederatedType(np.int32, placements.SERVER),
                computation_types.FederatedType([], placements.SERVER),
            ]),
        )
    )

  def test_init_does_not_raise_type_error_with_unknown_dimensions(self):
    state_type = computation_types.TensorType(np.int32, [None])

    @federated_computation.federated_computation(
        computation_types.FederatedType(state_type, placements.SERVER)
    )
    def server_prepare(server_state):
      return [[
          server_state,
      ]], [server_state]

    @federated_computation.federated_computation(
        [[computation_types.FederatedType(state_type, placements.SERVER)]]
    )
    def server_to_client_broadcast(context_at_server):
      return [intrinsics.federated_broadcast(context_at_server[0][0])]

    @federated_computation.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(np.float32), placements.CLIENTS
        ),
        [computation_types.FederatedType(state_type, placements.CLIENTS)],
    )
    def client_work(client_data, context_at_clients):
      @tensorflow_computation.tf_computation
      def client_work_tf():
        return 1

      del client_data  # Unused
      del context_at_clients  # Unused
      return [
          [intrinsics.federated_value(client_work_tf(), placements.CLIENTS)]
      ]

    @federated_computation.federated_computation(
        [computation_types.FederatedType(state_type, placements.SERVER)],
        [[computation_types.FederatedType(np.int32, placements.CLIENTS)]],
    )
    def client_to_server_aggregation(temp_server_state, client_updates):
      del temp_server_state  # Unused
      return [intrinsics.federated_sum(client_updates[0][0])]

    @federated_computation.federated_computation(
        [computation_types.FederatedType(state_type, placements.SERVER)],
        [computation_types.FederatedType(np.int32, placements.SERVER)],
    )
    def server_result(temp_server_state, aggregated_results):
      return temp_server_state[0], aggregated_results[0]

    type_signature = (
        distribute_aggregate_test_utils.generate_unnamed_type_signature(
            server_prepare, client_work, server_result
        )
    )
    try:
      forms.DistributeAggregateForm(
          type_signature,
          server_prepare,
          server_to_client_broadcast,
          client_work,
          client_to_server_aggregation,
          server_result,
      )
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_init_raises_type_error_with_bad_server_prepare_parameter_type(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.float32, placements.SERVER)
    )
    def server_prepare(server_state):
      del server_state  # Unused

      @tensorflow_computation.tf_computation
      def server_prepare_broadcast_tf():
        return 1.0

      @tensorflow_computation.tf_computation
      def server_prepare_state_tf():
        return 32

      return [
          [
              intrinsics.federated_value(
                  server_prepare_broadcast_tf(), placements.SERVER
              )
          ]
      ], [
          server_prepare_state_tf(),
          intrinsics.federated_value(
              server_prepare_state_tf(), placements.SERVER
          ),
      ]

    with self.assertRaisesRegex(
        TypeError, 'the `server_prepare` computation argument type'
    ):
      _build_test_distribute_aggregate_form_with_computations(
          server_prepare=server_prepare
      )

  def test_init_raises_type_error_with_broadcast_input_type_mismatch(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER)
    )
    def server_prepare(server_state):
      @tensorflow_computation.tf_computation
      def server_prepare_state_tf():
        return 32

      return [], [server_prepare_state_tf(), server_state]

    with self.assertRaisesRegex(
        TypeError,
        'The `server_to_client_broadcast` computation expects an argument type',
    ):
      _build_test_distribute_aggregate_form_with_computations(
          server_prepare=server_prepare
      )

  def test_init_raises_type_error_with_broadcast_output_type_mismatch(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(np.float32), placements.CLIENTS
        ),
        [computation_types.FederatedType(np.int32, placements.CLIENTS)],
    )
    def client_work(client_data, context_at_clients):
      @tensorflow_computation.tf_computation
      def client_work_tf():
        return [1, 2]

      del client_data  # Unused
      del context_at_clients  # Unused
      return [
          [intrinsics.federated_value(client_work_tf(), placements.CLIENTS)]
      ]

    with self.assertRaisesRegex(
        TypeError, 'The `client_work` computation expects an argument type'
    ):
      _build_test_distribute_aggregate_form_with_computations(
          client_work=client_work
      )

  def test_init_raises_assertion_error_with_bad_broadcast_body(self):
    @tensorflow_computation.tf_computation
    def multiply_tf(x):
      return x * 2.0

    @federated_computation.federated_computation(
        [[computation_types.FederatedType(np.float32, placements.SERVER)]]
    )
    def server_to_client_broadcast(context_at_server):
      a = intrinsics.federated_map(multiply_tf, context_at_server[0][0])
      return [intrinsics.federated_broadcast(a)]

    with self.assertRaisesRegex(
        ValueError, 'Expected only broadcast intrinsics'
    ):
      _build_test_distribute_aggregate_form_with_computations(
          server_to_client_broadcast=server_to_client_broadcast
      )

  def test_init_raises_type_error_with_aggregation_input_type_mismatch(self):

    @federated_computation.federated_computation(
        [
            np.int32,
            computation_types.FederatedType(np.int32, placements.SERVER),
        ],
        [[computation_types.FederatedType(np.int32, placements.CLIENTS)]],
    )
    def client_to_server_aggregation(temp_server_state, client_updates):
      del temp_server_state  # Unused.
      return [
          intrinsics.federated_secure_sum_bitwidth(client_updates[0][0], 100)
      ]

    with self.assertRaisesRegex(
        TypeError,
        (
            'The `client_to_server_aggregation` computation expects an argument'
            ' type'
        ),
    ):
      _build_test_distribute_aggregate_form_with_computations(
          client_to_server_aggregation=client_to_server_aggregation
      )

  def test_init_raises_type_error_with_aggregation_output_type_mismatch(self):

    @federated_computation.federated_computation(
        [
            np.int32,
            computation_types.FederatedType(np.int32, placements.SERVER),
        ],
        [[
            computation_types.FederatedType(
                computation_types.TensorType(np.int32, [2]), placements.CLIENTS
            )
        ]],
    )
    def client_to_server_aggregation(temp_server_state, client_updates):
      del temp_server_state  # Unused
      a = intrinsics.federated_sum(client_updates[0][0])
      b = intrinsics.federated_sum(client_updates[0][0])
      return [a, b]

    with self.assertRaisesRegex(
        TypeError, 'The `server_result` computation expects an argument type'
    ):
      _build_test_distribute_aggregate_form_with_computations(
          client_to_server_aggregation=client_to_server_aggregation
      )

  def test_init_raises_assertion_error_with_bad_aggregation_body(self):

    @federated_computation.federated_computation(
        [
            np.int32,
            computation_types.FederatedType(np.int32, placements.SERVER),
        ],
        [[
            computation_types.FederatedType(
                computation_types.TensorType(np.int32, [2]), placements.CLIENTS
            )
        ]],
    )
    def client_to_server_aggregation(temp_server_state, client_updates):
      del temp_server_state  # Unused.
      a = intrinsics.federated_secure_sum_bitwidth(client_updates[0][0], 100)
      b = intrinsics.federated_sum(client_updates[0][0])
      return [b, a]

    @federated_computation.federated_computation(
        [
            np.int32,
            computation_types.FederatedType(np.int32, placements.SERVER),
        ],
        [
            computation_types.FederatedType(
                computation_types.TensorType(np.int32, [2]), placements.SERVER
            ),
            computation_types.FederatedType(
                computation_types.TensorType(np.int32, [2]), placements.SERVER
            ),
        ],
    )
    def server_result(temp_server_state, aggregated_results):
      del aggregated_results  # Unused
      return temp_server_state[1], intrinsics.federated_value(
          [], placements.SERVER
      )

    with self.assertRaisesRegex(
        ValueError, 'Expected the aggregation function to return references'
    ):
      _build_test_distribute_aggregate_form_with_computations(
          client_to_server_aggregation=client_to_server_aggregation,
          server_result=server_result,
      )

  def test_init_raises_type_error_with_temporary_state_type_mismatch(self):

    @federated_computation.federated_computation(
        [
            np.int32,
            computation_types.FederatedType(np.int32, placements.SERVER),
            np.int32,
        ],
        [[
            computation_types.FederatedType(
                computation_types.TensorType(np.int32, [2]), placements.CLIENTS
            )
        ]],
    )
    def client_to_server_aggregation(temp_server_state, client_updates):
      del temp_server_state  # Unused.
      return [
          intrinsics.federated_secure_sum_bitwidth(client_updates[0][0], 100)
      ]

    @federated_computation.federated_computation(
        [
            np.int32,
            computation_types.FederatedType(np.int32, placements.SERVER),
            np.int32,
        ],
        [
            computation_types.FederatedType(
                computation_types.TensorType(np.int32, [2]), placements.SERVER
            )
        ],
    )
    def server_result(temp_server_state, aggregated_results):
      del aggregated_results  # Unused
      return temp_server_state[1], intrinsics.federated_value(
          [], placements.SERVER
      )

    with self.assertRaisesRegex(
        TypeError,
        (
            'The `client_to_server_aggregation` computation expects an argument'
            ' type'
        ),
    ):
      _build_test_distribute_aggregate_form_with_computations(
          client_to_server_aggregation=client_to_server_aggregation
      )

    with self.assertRaisesRegex(
        TypeError, 'the `server_result` computation expects an argument type'
    ):
      _build_test_distribute_aggregate_form_with_computations(
          server_result=server_result
      )

  def test_init_raises_type_error_with_type_signature_mismatch(self):
    (test_server_prepare, _, test_client_work, _, test_server_result) = (
        _test_distribute_aggregate_form_computations()
    )
    correct_type_signature = (
        distribute_aggregate_test_utils.generate_unnamed_type_signature(
            test_server_prepare, test_client_work, test_server_result
        )
    )

    bad_server_state_parameter = computation_types.StructType([
        computation_types.FederatedType(np.float32, placements.SERVER),
        test_client_work.type_signature.parameter[0],
    ])
    bad_client_data_parameter = computation_types.StructType([
        test_server_result.type_signature.parameter[0],
        computation_types.FederatedType(np.str_, placements.CLIENTS),
    ])

    bad_server_state_result = computation_types.StructType([
        computation_types.FederatedType(np.float32, placements.SERVER),
        test_server_result.type_signature.result[1],
    ])

    bad_server_output_result = computation_types.StructType([
        test_server_result.type_signature.parameter[0],
        computation_types.FederatedType(np.float32, placements.SERVER),
    ])

    with self.assertRaisesRegex(
        TypeError, 'The original computation argument type'
    ):
      _build_test_distribute_aggregate_form_with_computations(
          type_signature=computation_types.FunctionType(
              bad_server_state_parameter, correct_type_signature.result
          )
      )

    with self.assertRaisesRegex(
        TypeError, 'The original computation argument type'
    ):
      _build_test_distribute_aggregate_form_with_computations(
          type_signature=computation_types.FunctionType(
              bad_client_data_parameter, correct_type_signature.result
          )
      )

    with self.assertRaisesRegex(
        TypeError, 'the original computation result type'
    ):
      _build_test_distribute_aggregate_form_with_computations(
          type_signature=computation_types.FunctionType(
              correct_type_signature.parameter, bad_server_state_result
          )
      )

    with self.assertRaisesRegex(
        TypeError, 'the original computation result type'
    ):
      _build_test_distribute_aggregate_form_with_computations(
          type_signature=computation_types.FunctionType(
              correct_type_signature.parameter, bad_server_output_result
          )
      )

  def test_summary(self):
    daf = distribute_aggregate_test_utils.get_temperature_sensor_example().daf

    class CapturePrint:

      def __init__(self):
        self.summary = ''

      def __call__(self, msg):
        self.summary += msg + '\n'

    capture = CapturePrint()
    daf.summary(print_fn=capture)
    # pyformat: disable
    self.assertEqual(
        capture.summary,
        'server_prepare              : (<num_rounds=int32>@SERVER -> <<<max_temperature=float32>@SERVER>,<<num_rounds=int32>@SERVER>>)\n'
        'server_to_client_broadcast  : (<<max_temperature=float32>@SERVER> -> <<max_temperature=float32>@CLIENTS>)\n'
        'client_work                 : (<data={float32*}@CLIENTS,context_at_client=<{<max_temperature=float32>}@CLIENTS>> -> <is_over={float32}@CLIENTS,weight={float32}@CLIENTS>)\n'
        'client_to_server_aggregation: (<intermediate_server_state=<<num_rounds=int32>@SERVER>,client_updates=<is_over={float32}@CLIENTS,weight={float32}@CLIENTS>> -> <float32@SERVER>)\n'
        'server_result               : (<intermediate_server_state=<<num_rounds=int32>@SERVER>,aggregation_result=<float32@SERVER>> -> <<num_rounds=int32>@SERVER,<ratio_over_threshold=float32@SERVER>>)\n'
    )
    # pyformat: enable


if __name__ == '__main__':
  absltest.main()
