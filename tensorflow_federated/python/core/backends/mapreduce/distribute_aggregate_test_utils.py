# Copyright 2022, The TensorFlow Federated Authors.
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
"""Utilities for testing the DistributeAggregateForm backend."""

import collections

import federated_language
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.mapreduce import forms
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.compiler import tree_transformations

DistributeAggregateFormExample = collections.namedtuple(
    'DistributeAggregateFormExample', ['daf', 'initialize']
)


def generate_unnamed_type_signature(
    server_prepare: federated_language.framework.ConcreteComputation,
    client_work: federated_language.framework.ConcreteComputation,
    server_result: federated_language.framework.ConcreteComputation,
) -> federated_language.FunctionType:
  """Generates a type signature for the DistributeAggregateForm."""
  parameter = federated_language.StructType([
      server_prepare.type_signature.parameter,
      client_work.type_signature.parameter[0],  # pytype: disable=unsupported-operands
  ])
  result = server_result.type_signature.result
  return federated_language.FunctionType(parameter, result)


def _make_distribute_aggregate_form_example(
    initialize: federated_language.framework.ConcreteComputation,
    type_signature: federated_language.FunctionType,
    server_prepare: federated_language.framework.ConcreteComputation,
    server_to_client_broadcast: federated_language.framework.ConcreteComputation,
    client_work: federated_language.framework.ConcreteComputation,
    client_to_server_aggregation: federated_language.framework.ConcreteComputation,
    server_result: federated_language.framework.ConcreteComputation,
) -> DistributeAggregateFormExample:
  """Constructs a DistributeAggregateFormExample given the component comps."""

  def _uniquify_reference_names(
      comp: federated_language.framework.ConcreteComputation,
  ):
    building_block = comp.to_building_block()
    transformed_comp = tree_transformations.uniquify_reference_names(
        building_block
    )[0]
    return federated_language.framework.ConcreteComputation(
        computation_proto=transformed_comp.proto,
        context_stack=federated_language.framework.get_context_stack(),
    )

  return DistributeAggregateFormExample(
      daf=forms.DistributeAggregateForm(
          type_signature,
          _uniquify_reference_names(server_prepare),
          _uniquify_reference_names(server_to_client_broadcast),
          _uniquify_reference_names(client_work),
          _uniquify_reference_names(client_to_server_aggregation),
          _uniquify_reference_names(server_result),
      ),
      initialize=_uniquify_reference_names(initialize),
  )


def get_temperature_sensor_example() -> DistributeAggregateFormExample:
  """Constructs `forms.DistributeAggregateForm` for temperature sensors example.

  The temperature sensor example computes the fraction of sensors that report
  temperatures over the threshold.

  Returns:
    A tuple of: (1) an instance of `forms.DistributeAggregateForm` and (2) an
    associated `federated_language.framework.Computation` that generates an
    initial state
    compatible with the server state expected by the
    `forms.DistributeAggregateForm`.
  """

  @federated_language.federated_computation()
  def initialize():
    @tensorflow_computation.tf_computation
    def initialize_tf():
      return collections.OrderedDict(num_rounds=0)

    return federated_language.federated_value(
        initialize_tf(), federated_language.SERVER
    )

  # The state of the server is a struct containing just the integer
  # counter `num_rounds`.
  server_state_type = [('num_rounds', np.int32)]

  @federated_language.federated_computation(
      federated_language.FederatedType(
          server_state_type, federated_language.SERVER
      )
  )
  def server_prepare(state):
    @tensorflow_computation.tf_computation(server_state_type)
    def server_prepare_tf(state):
      return collections.OrderedDict(
          max_temperature=32.0 + tf.cast(state['num_rounds'], tf.float32)
      )

    broadcast_args = [
        federated_language.federated_map(server_prepare_tf, state)
    ]
    intermediate_state = [state]
    return broadcast_args, intermediate_state

  # The broadcast input and output types are both structs containing a single
  # float `max_temperature`, which is the threshold received from the server.
  broadcast_type = collections.OrderedDict(max_temperature=np.float32)

  # The intermediate state will contain the server state.
  intermediate_state_type = [
      federated_language.FederatedType(
          server_state_type, federated_language.SERVER
      )
  ]

  @federated_language.federated_computation([
      federated_language.FederatedType(
          broadcast_type, federated_language.SERVER
      )
  ])
  def server_to_client_broadcast(context_at_server):
    return [federated_language.federated_broadcast(context_at_server[0])]

  # The client data is a sequence of floats.
  client_data_type = federated_language.SequenceType(np.float32)

  @federated_language.federated_computation(
      federated_language.FederatedType(
          client_data_type, federated_language.CLIENTS
      ),
      [
          federated_language.FederatedType(
              broadcast_type, federated_language.CLIENTS
          )
      ],
  )
  def client_work(data, context_at_client):
    @tensorflow_computation.tf_computation(client_data_type, [broadcast_type])
    def client_work_tf(data_tf, context_at_client_tf):
      def fn(s, x):
        return {
            'num': s['num'] + 1,
            'max': tf.maximum(s['max'], x),
        }

      reduce_result = data_tf.reduce(
          {'num': np.int32(0), 'max': np.float32(-459.67)}, fn
      )
      client_updates = collections.OrderedDict(
          is_over=tf.cast(
              reduce_result['max'] > context_at_client_tf[0]['max_temperature'],
              tf.float32,
          ),
          weight=1.0,
      )
      return client_updates

    results = federated_language.federated_map(
        client_work_tf, (data, context_at_client)
    )
    unzipped_results = federated_language.framework.create_federated_unzip(
        results.comp
    )
    return federated_language.Value(
        federated_language.framework.get_context_stack().current.bind_computation_to_reference(
            unzipped_results
        )
    )

  # The client update is a struct.
  federated_client_update_type = [
      (
          'is_over',
          federated_language.FederatedType(
              np.float32, federated_language.CLIENTS
          ),
      ),
      (
          'weight',
          federated_language.FederatedType(
              np.float32, federated_language.CLIENTS
          ),
      ),
  ]

  @federated_language.federated_computation(
      intermediate_state_type, federated_client_update_type
  )
  def client_to_server_aggregation(intermediate_server_state, client_updates):
    del intermediate_server_state  # Unused
    return [
        federated_language.federated_mean(client_updates[0], client_updates[1])
    ]

  # The aggregation result type is a single float.
  aggregation_result_type = np.float32

  @federated_language.federated_computation(
      intermediate_state_type,
      [
          federated_language.FederatedType(
              aggregation_result_type, federated_language.SERVER
          )
      ],
  )
  def server_result(intermediate_server_state, aggregation_result):
    @tensorflow_computation.tf_computation(server_state_type)
    def server_result_tf(server_state):
      return collections.OrderedDict(num_rounds=server_state['num_rounds'] + 1)

    return federated_language.federated_map(
        server_result_tf, intermediate_server_state[0]
    ), collections.OrderedDict(ratio_over_threshold=aggregation_result[0])

  type_signature = generate_unnamed_type_signature(
      server_prepare, client_work, server_result
  )
  return _make_distribute_aggregate_form_example(
      initialize,
      type_signature,
      server_prepare,
      server_to_client_broadcast,
      client_work,
      client_to_server_aggregation,
      server_result,
  )


def get_mnist_training_example() -> DistributeAggregateFormExample:
  """Constructs `forms.DistributeAggregateForm` for mnist training.

  Returns:
    An instance of `forms.DistributeAggregateForm`.
  """
  model_nt = collections.namedtuple('Model', 'weights bias')
  server_state_nt = collections.namedtuple('ServerState', 'model num_rounds')

  # Start with a model filled with zeros, and the round counter set to zero.
  @federated_language.federated_computation()
  def initialize():
    @tensorflow_computation.tf_computation
    def initialize_tf():
      return server_state_nt(
          model=model_nt(weights=np.zeros([784, 10]), bias=np.zeros([10])),
          num_rounds=0,
      )

    return federated_language.federated_value(
        initialize_tf(), federated_language.SERVER
    )

  server_state_tff_type = server_state_nt(
      model=model_nt(weights=(np.float32, [784, 10]), bias=(np.float32, [10])),
      num_rounds=np.int32,
  )
  client_state_nt = collections.namedtuple('ClientState', 'model learning_rate')

  # Prepare the broadcast input containing the model and a dynamically adjusted
  # learning rate that starts at 0.1 and decays exponentially by a factor of
  # 0.9.
  @federated_language.federated_computation(
      federated_language.FederatedType(
          server_state_tff_type, federated_language.SERVER
      )
  )
  def server_prepare(state):
    @tensorflow_computation.tf_computation(server_state_tff_type)
    def server_prepare_tf(state):
      learning_rate = 0.1 * tf.pow(0.9, tf.cast(state.num_rounds, tf.float32))
      return client_state_nt(model=state.model, learning_rate=learning_rate)

    broadcast_args = [
        federated_language.federated_map(server_prepare_tf, state)
    ]
    intermediate_state = [32, state]
    return broadcast_args, intermediate_state

  # The intermediate state is a struct containiing the bitwidth for the secure
  # sum and the server state.
  intermediate_state_type = [
      np.int32,
      federated_language.FederatedType(
          server_state_tff_type, federated_language.SERVER
      ),
  ]

  model_tff_type = model_nt(
      weights=(np.float32, [784, 10]), bias=(np.float32, [10])
  )
  broadcast_tff_type = client_state_nt(
      model=model_tff_type, learning_rate=np.float32
  )

  @federated_language.federated_computation([
      federated_language.FederatedType(
          broadcast_tff_type, federated_language.SERVER
      )
  ])
  def server_to_client_broadcast(context_at_server):
    return [federated_language.federated_broadcast(context_at_server[0])]

  batch_nt = collections.namedtuple('Batch', 'x y')
  batch_tff_type = batch_nt(x=(np.float32, [None, 784]), y=(np.int32, [None]))
  dataset_tff_type = federated_language.SequenceType(batch_tff_type)
  loop_state_nt = collections.namedtuple('LoopState', 'num_examples total_loss')
  update_nt = collections.namedtuple('Update', 'model num_examples loss')

  # Train the model locally, emit the locally-trained model and the number of
  # examples as an update, and the average loss and the number of examples as
  # local client stats.
  @federated_language.federated_computation(
      federated_language.FederatedType(
          dataset_tff_type, federated_language.CLIENTS
      ),
      [
          federated_language.FederatedType(
              broadcast_tff_type, federated_language.CLIENTS
          )
      ],
  )
  def client_work(data, context_at_client):
    @tensorflow_computation.tf_computation(dataset_tff_type, broadcast_tff_type)
    def client_work_tf(data, context_at_client):
      model_vars = model_nt(
          weights=tf.Variable(
              initial_value=context_at_client.model.weights, name='weights'
          ),
          bias=tf.Variable(
              initial_value=context_at_client.model.bias, name='bias'
          ),
      )
      init_model = tf.compat.v1.global_variables_initializer()

      optimizer = tf.keras.optimizers.SGD(context_at_client.learning_rate)

      @tf.function
      def reduce_fn(loop_state, batch):
        """Compute a single gradient step on a given batch of examples."""
        with tf.GradientTape() as tape:
          pred_y = tf.nn.softmax(
              tf.matmul(batch.x, model_vars.weights) + model_vars.bias
          )
          loss = -tf.reduce_mean(
              tf.reduce_sum(
                  tf.one_hot(batch.y, 10) * tf.math.log(pred_y), axis=[1]
              )
          )
        grads = tape.gradient(loss, model_vars)
        optimizer.apply_gradients(
            zip(tf.nest.flatten(grads), tf.nest.flatten(model_vars))
        )
        return loop_state_nt(
            num_examples=loop_state.num_examples + 1,
            total_loss=loop_state.total_loss + loss,
        )

      with tf.control_dependencies([init_model]):
        loop_state = data.reduce(
            loop_state_nt(num_examples=0, total_loss=np.float32(0.0)), reduce_fn
        )
        num_examples = loop_state.num_examples
        total_loss = loop_state.total_loss
        with tf.control_dependencies([num_examples, total_loss]):
          loss = total_loss / tf.cast(num_examples, tf.float32)

      return update_nt(model=model_vars, num_examples=num_examples, loss=loss)

    @tensorflow_computation.tf_computation(np.int32)
    def cast_to_float(val):
      return tf.cast(val, tf.float32)

    results = federated_language.federated_map(
        client_work_tf,
        federated_language.federated_zip([data, context_at_client[0]]),
    )
    unzipped_results = federated_language.framework.create_federated_unzip(
        results.comp
    )
    client_update = federated_language.Value(
        federated_language.framework.get_context_stack().current.bind_computation_to_reference(
            unzipped_results
        )
    )

    return [
        # input for federated_mean
        client_update.model,
        federated_language.federated_map(
            cast_to_float, client_update.num_examples
        ),
        # input for federated_sum
        client_update.num_examples,
        # input for federated_mean
        client_update.loss,
        federated_language.federated_map(
            cast_to_float, client_update.num_examples
        ),
    ]

  federated_aggregation_input_tff_type = [
      # input for federated_mean
      federated_language.FederatedType(
          model_tff_type, federated_language.CLIENTS
      ),
      federated_language.FederatedType(np.float32, federated_language.CLIENTS),
      # input for federated_sum
      federated_language.FederatedType(np.int32, federated_language.CLIENTS),
      # input for federated_mean
      federated_language.FederatedType(np.float32, federated_language.CLIENTS),
      federated_language.FederatedType(np.float32, federated_language.CLIENTS),
  ]

  @federated_language.federated_computation(
      intermediate_state_type, federated_aggregation_input_tff_type
  )
  def client_to_server_aggregation(intermediate_server_state, client_updates):
    scaled_model = federated_language.federated_mean(
        client_updates[0], client_updates[1]
    )
    num_examples = federated_language.federated_secure_sum_bitwidth(
        client_updates[2], intermediate_server_state[0]
    )

    scaled_loss = federated_language.federated_mean(
        client_updates[3], client_updates[4]
    )
    return [scaled_model, num_examples, scaled_loss]

  # The aggregation result type is a struct.
  federated_aggregation_result_type = update_nt(
      model=federated_language.FederatedType(
          model_tff_type, federated_language.SERVER
      ),
      num_examples=federated_language.FederatedType(
          np.int32, federated_language.SERVER
      ),
      loss=federated_language.FederatedType(
          np.float32, federated_language.SERVER
      ),
  )

  metrics_nt = collections.namedtuple('Metrics', 'num_rounds num_examples loss')

  @federated_language.federated_computation(
      intermediate_state_type,
      federated_aggregation_result_type,
  )
  def server_result(intermediate_server_state, aggregation_result):
    @tensorflow_computation.tf_computation(server_state_tff_type)
    def server_result_tf(state):
      return state.num_rounds + 1

    num_rounds = federated_language.federated_map(
        server_result_tf, intermediate_server_state[1]
    )

    new_server_state = federated_language.federated_zip(
        server_state_nt(model=aggregation_result.model, num_rounds=num_rounds)
    )
    metrics = federated_language.federated_zip(
        metrics_nt(
            num_rounds=num_rounds,
            num_examples=aggregation_result.num_examples,
            loss=aggregation_result.loss,
        )
    )
    return new_server_state, metrics

  type_signature = generate_unnamed_type_signature(
      server_prepare, client_work, server_result
  )
  return _make_distribute_aggregate_form_example(
      initialize,
      type_signature,
      server_prepare,
      server_to_client_broadcast,
      client_work,
      client_to_server_aggregation,
      server_result,
  )
