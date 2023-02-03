_-$
./
".GO"
/Comparing changes

Choose two branches to see what’s changed or to start a new pull request. If you need to, you can also .

 

 

...

 

 

  Able to merge. These branches can be automatically merged.

 create Build.go to run.build #3599

No description available

 2 commits

 2 files changed

 1 contributor

Commits on Feb 3, 2023

Rename BUILD to Build.go

@MoneyMan573

MoneyMan573 committed 10 minutes ago

  

Update Build.go

@MoneyMan573

MoneyMan573 committed 4 minutes ago

  

Showing  with 10 additions and 5 deletions.

 5  

BUILD

@@ -1,5 +0,0 @@

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

 10  

Build.go

@@ -0,0 +1,10 @@

_-$

./

".GO"

/

.BUILD.GO/.main-tensorflow-federated.config.json

Filter changed files
  12  
tensorflow_federated/python/core/backends/mapreduce/BUILD
@@ -121,6 +121,8 @@ py_library(
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/compiler:transformations",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
@@ -140,6 +142,7 @@ py_test(
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":distribute_aggregate_test_utils",
        ":form_utils",
        ":forms",
        ":mapreduce_test_utils",
@@ -165,8 +168,17 @@ py_library(
    srcs = ["distribute_aggregate_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":forms",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/federated_context:value_impl",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

  2  
tensorflow_federated/python/core/backends/mapreduce/__init__.py
@@ -293,7 +293,7 @@ def round_comp(server_state, client_data):
from tensorflow_federated.python.core.backends.mapreduce.form_utils import get_computation_for_broadcast_form
from tensorflow_federated.python.core.backends.mapreduce.form_utils import get_computation_for_map_reduce_form
from tensorflow_federated.python.core.backends.mapreduce.form_utils import get_map_reduce_form_for_computation
from tensorflow_federated.python.core.backends.mapreduce.form_utils import get_state_initialization_computation_for_map_reduce_form
from tensorflow_federated.python.core.backends.mapreduce.form_utils import get_state_initialization_computation
from tensorflow_federated.python.core.backends.mapreduce.forms import BroadcastForm
from tensorflow_federated.python.core.backends.mapreduce.forms import DistributeAggregateForm
from tensorflow_federated.python.core.backends.mapreduce.forms import MapReduceForm
  16  
tensorflow_federated/python/core/backends/mapreduce/compiler_test.py
@@ -94,9 +94,7 @@ def test_raises_non_function_and_compiled_computation(self):
    initialize = (
        mapreduce_test_utils.get_temperature_sensor_example().initialize
    )
    init = form_utils.get_state_initialization_computation_for_map_reduce_form(
        initialize
    )
    init = form_utils.get_state_initialization_computation(initialize)
    compiled_computation = self.compiled_computation_for_initialize(init)
    integer_ref = building_blocks.Reference('x', tf.int32)
    with self.assertRaisesRegex(
@@ -109,9 +107,7 @@ def test_raises_function_and_compiled_computation_of_different_type(self):
    initialize = (
        mapreduce_test_utils.get_temperature_sensor_example().initialize
    )
    init = form_utils.get_state_initialization_computation_for_map_reduce_form(
        initialize
    )
    init = form_utils.get_state_initialization_computation(initialize)
    compiled_computation = self.compiled_computation_for_initialize(init)
    function = building_blocks.Reference(
        'f', computation_types.FunctionType(tf.int32, tf.int32)
@@ -136,9 +132,7 @@ def test_passes_function_and_compiled_computation_of_same_type(self):
    initialize = (
        mapreduce_test_utils.get_temperature_sensor_example().initialize
    )
    init = form_utils.get_state_initialization_computation_for_map_reduce_form(
        initialize
    )
    init = form_utils.get_state_initialization_computation(initialize)
    compiled_computation = self.compiled_computation_for_initialize(init)
    function = building_blocks.Reference(
        'f', compiled_computation.type_signature
@@ -158,9 +152,7 @@ def test_already_reduced_case(self):
    initialize = (
        mapreduce_test_utils.get_temperature_sensor_example().initialize
    )
    init = form_utils.get_state_initialization_computation_for_map_reduce_form(
        initialize
    )
    init = form_utils.get_state_initialization_computation(initialize)

    comp = init.to_building_block()

 409  
tensorflow_federated/python/core/backends/mapreduce/distribute_aggregate_test_utils.py
@@ -13,8 +13,26 @@
# limitations under the License.
"""Utilities for testing the DistributeAggregateForm backend."""

import collections

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.mapreduce import forms
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements

DistributeAggregateFormExample = collections.namedtuple(
    'DistributeAggregateFormExample', ['daf', 'initialize']
)


def generate_unnamed_type_signature(
@@ -29,3 +47,394 @@ def generate_unnamed_type_signature(
  ])
  result = server_result.type_signature.result
  return computation_types.FunctionType(parameter, result)


def _make_distribute_aggregate_form_example(
    initialize: computation_impl.ConcreteComputation,
    type_signature: computation_types.FunctionType,
    server_prepare: computation_impl.ConcreteComputation,
    server_to_client_broadcast: computation_impl.ConcreteComputation,
    client_work: computation_impl.ConcreteComputation,
    client_to_server_aggregation: computation_impl.ConcreteComputation,
    server_result: computation_impl.ConcreteComputation,
) -> DistributeAggregateFormExample:
  """Constructs a DistributeAggregateFormExample given the component comps."""

  def _uniquify_reference_names(comp: computation_impl.ConcreteComputation):
    return computation_impl.ConcreteComputation.from_building_block(
        tree_transformations.uniquify_reference_names(comp.to_building_block())[
            0
        ]
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
    associated `computation_base.Computation` that generates an initial state
    compatible with the server state expected by the
    `forms.DistributeAggregateForm`.
  """

  @federated_computation.federated_computation()
  def initialize():
    @tensorflow_computation.tf_computation
    def initialize_tf():
      return collections.OrderedDict(num_rounds=tf.constant(0))

    return intrinsics.federated_value(initialize_tf(), placements.SERVER)

  # The state of the server is a struct containing just the integer
  # counter `num_rounds`.
  server_state_type = [('num_rounds', tf.int32)]

  @federated_computation.federated_computation(
      computation_types.at_server(server_state_type)
  )
  def server_prepare(state):
    @tensorflow_computation.tf_computation(server_state_type)
    def server_prepare_tf(state):
      return collections.OrderedDict(
          max_temperature=32.0 + tf.cast(state['num_rounds'], tf.float32)
      )

    broadcast_args = [intrinsics.federated_map(server_prepare_tf, state)]
    intermediate_state = [state]
    return broadcast_args, intermediate_state

  # The broadcast input and output types are both structs containing a single
  # float `max_temperature`, which is the threshold received from the server.
  broadcast_type = collections.OrderedDict(max_temperature=tf.float32)

  # The intermediate state will contain the server state.
  intermediate_state_type = [computation_types.at_server(server_state_type)]

  @federated_computation.federated_computation(
      [computation_types.at_server(broadcast_type)]
  )
  def server_to_client_broadcast(context_at_server):
    return [intrinsics.federated_broadcast(context_at_server[0])]

  # The client data is a sequence of floats.
  client_data_type = computation_types.SequenceType(tf.float32)

  @federated_computation.federated_computation(
      computation_types.at_clients(client_data_type),
      [computation_types.at_clients(broadcast_type)],
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

    results = intrinsics.federated_map(
        client_work_tf, (data, context_at_client)
    )
    unzipped_results = building_block_factory.create_federated_unzip(
        results.comp
    )
    return value_impl.Value(
        context_stack_impl.context_stack.current.bind_computation_to_reference(
            unzipped_results
        )
    )

  # The client update is a struct.
  federated_client_update_type = [
      ('is_over', computation_types.at_clients(tf.float32)),
      ('weight', computation_types.at_clients(tf.float32)),
  ]

  @federated_computation.federated_computation(
      intermediate_state_type, federated_client_update_type
  )
  def client_to_server_aggregation(intermediate_server_state, client_updates):
    del intermediate_server_state  # Unused
    return [intrinsics.federated_mean(client_updates[0], client_updates[1])]

  # The aggregation result type is a single float.
  aggregation_result_type = tf.float32

  @federated_computation.federated_computation(
      intermediate_state_type,
      [computation_types.at_server(aggregation_result_type)],
  )
  def server_result(intermediate_server_state, aggregation_result):
    @tensorflow_computation.tf_computation(server_state_type)
    def server_result_tf(server_state):
      return collections.OrderedDict(num_rounds=server_state['num_rounds'] + 1)

    return intrinsics.federated_map(
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
  @federated_computation.federated_computation()
  def initialize():
    @tensorflow_computation.tf_computation
    def initialize_tf():
      return server_state_nt(
          model=model_nt(weights=tf.zeros([784, 10]), bias=tf.zeros([10])),
          num_rounds=tf.constant(0),
      )

    return intrinsics.federated_value(initialize_tf(), placements.SERVER)

  server_state_tff_type = server_state_nt(
      model=model_nt(weights=(tf.float32, [784, 10]), bias=(tf.float32, [10])),
      num_rounds=tf.int32,
  )
  client_state_nt = collections.namedtuple('ClientState', 'model learning_rate')

  # Prepare the broadcast input containing the model and a dynamically adjusted
  # learning rate that starts at 0.1 and decays exponentially by a factor of
  # 0.9.
  @federated_computation.federated_computation(
      computation_types.at_server(server_state_tff_type)
  )
  def server_prepare(state):
    @tensorflow_computation.tf_computation(server_state_tff_type)
    def server_prepare_tf(state):
      learning_rate = 0.1 * tf.pow(0.9, tf.cast(state.num_rounds, tf.float32))
      return client_state_nt(model=state.model, learning_rate=learning_rate)

    broadcast_args = [intrinsics.federated_map(server_prepare_tf, state)]
    intermediate_state = [32, state]
    return broadcast_args, intermediate_state

  # The intermediate state is a struct containiing the bitwidth for the secure
  # sum and the server state.
  intermediate_state_type = [
      tf.int32,
      computation_types.at_server(server_state_tff_type),
  ]

  model_tff_type = model_nt(
      weights=(tf.float32, [784, 10]), bias=(tf.float32, [10])
  )
  broadcast_tff_type = client_state_nt(
      model=model_tff_type, learning_rate=tf.float32
  )

  @federated_computation.federated_computation(
      [computation_types.at_server(broadcast_tff_type)]
  )
  def server_to_client_broadcast(context_at_server):
    return [intrinsics.federated_broadcast(context_at_server[0])]

  batch_nt = collections.namedtuple('Batch', 'x y')
  batch_tff_type = batch_nt(x=(tf.float32, [None, 784]), y=(tf.int32, [None]))
  dataset_tff_type = computation_types.SequenceType(batch_tff_type)
  loop_state_nt = collections.namedtuple('LoopState', 'num_examples total_loss')
  update_nt = collections.namedtuple('Update', 'model num_examples loss')

  # Train the model locally, emit the locally-trained model and the number of
  # examples as an update, and the average loss and the number of examples as
  # local client stats.
  @federated_computation.federated_computation(
      computation_types.at_clients(dataset_tff_type),
      [computation_types.at_clients(broadcast_tff_type)],
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

    @tensorflow_computation.tf_computation(tf.int32)
    def cast_to_float(val):
      return tf.cast(val, tf.float32)

    results = intrinsics.federated_map(
        client_work_tf, intrinsics.federated_zip([data, context_at_client[0]])
    )
    unzipped_results = building_block_factory.create_federated_unzip(
        results.comp
    )
    client_update = value_impl.Value(
        context_stack_impl.context_stack.current.bind_computation_to_reference(
            unzipped_results
        )
    )

    return [
        # input for federated_mean
        client_update.model,
        intrinsics.federated_map(cast_to_float, client_update.num_examples),
        # input for federated_sum
        client_update.num_examples,
        # input for federated_mean
        client_update.loss,
        intrinsics.federated_map(cast_to_float, client_update.num_examples),
    ]

  federated_aggregation_input_tff_type = [
      # input for federated_mean
      computation_types.at_clients(model_tff_type),
      computation_types.at_clients(tf.float32),
      # input for federated_sum
      computation_types.at_clients(tf.int32),
      # input for federated_mean
      computation_types.at_clients(tf.float32),
      computation_types.at_clients(tf.float32),
  ]

  @federated_computation.federated_computation(
      intermediate_state_type, federated_aggregation_input_tff_type
  )
  def client_to_server_aggregation(intermediate_server_state, client_updates):
    scaled_model = intrinsics.federated_mean(
        client_updates[0], client_updates[1]
    )
    num_examples = intrinsics.federated_secure_sum_bitwidth(
        client_updates[2], intermediate_server_state[0]
    )

    scaled_loss = intrinsics.federated_mean(
        client_updates[3], client_updates[4]
    )
    return [scaled_model, num_examples, scaled_loss]

  # The aggregation result type is a struct.
  federated_aggregation_result_type = update_nt(
      model=computation_types.at_server(model_tff_type),
      num_examples=computation_types.at_server(tf.int32),
      loss=computation_types.at_server(tf.float32),
  )

  metrics_nt = collections.namedtuple('Metrics', 'num_rounds num_examples loss')

  @federated_computation.federated_computation(
      intermediate_state_type,
      federated_aggregation_result_type,
  )
  def server_result(intermediate_server_state, aggregation_result):
    @tensorflow_computation.tf_computation(server_state_tff_type)
    def server_result_tf(state):
      return state.num_rounds + 1

    num_rounds = intrinsics.federated_map(
        server_result_tf, intermediate_server_state[1]
    )

    new_server_state = intrinsics.federated_zip(
        server_state_nt(model=aggregation_result.model, num_rounds=num_rounds)
    )
    metrics = intrinsics.federated_zip(
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
 359  
tensorflow_federated/python/core/backends/mapreduce/form_utils.py
@@ -28,6 +28,8 @@
from tensorflow_federated.python.core.backends.mapreduce import forms
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
@@ -89,27 +91,26 @@ def computation(arg):
  return computation


def get_state_initialization_computation_for_map_reduce_form(
def get_state_initialization_computation(
    initialize_computation: computation_impl.ConcreteComputation,
    grappler_config: tf.compat.v1.ConfigProto = _GRAPPLER_DEFAULT_CONFIG,
) -> computation_base.Computation:
  """Validates and transforms a computation to generate state for MapReduceForm.
  """Validates and transforms a computation to generate state.
  Args:
    initialize_computation: A `computation_impl.ConcreteComputation` that should
      generate initial state for a computation that is compatible with
      MapReduceForm.
      generate initial state for a computation that is compatible with a
      federated learning system that implements the contract of a backend
      defined in the backends/mapreduce directory.
    grappler_config: An optional instance of `tf.compat.v1.ConfigProto` to
      configure Grappler graph optimization of the TensorFlow graphs backing the
      resulting `tff.backends.mapreduce.MapReduceForm`. These options are
      combined with a set of defaults that aggressively configure Grappler. If
      the input `grappler_config` has
      configure Grappler graph optimization of the TensorFlow graphs. These
      options are combined with a set of defaults that aggressively configure
      Grappler. If the input `grappler_config` has
      `graph_options.rewrite_options.disable_meta_optimizer=True`, Grappler is
      bypassed.
  Returns:
    A `computation_base.Computation` that can generate state for a computation
    that is compatible with MapReduceForm.
    A `computation_base.Computation` that can generate state for a computation.
  Raises:
    TypeError: If the arguments are of the wrong types.
@@ -196,6 +197,40 @@ def computation(arg):
  return computation


def get_computation_for_distribute_aggregate_form(
    daf: forms.DistributeAggregateForm,
) -> computation_base.Computation:
  """Creates `tff.Computation` from a DistributeAggregate form.
  Args:
    daf: An instance of `tff.backends.mapreduce.DistributeAggregateForm`.
  Returns:
    An instance of `tff.Computation` that corresponds to `daf`.
  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(daf, forms.DistributeAggregateForm)

  @federated_computation.federated_computation(daf.type_signature.parameter)
  def computation(arg):
    """The logic of a single federated computation round."""
    server_state, client_data = arg
    broadcast_input, temp_server_state = daf.server_prepare(server_state)
    broadcast_output = daf.server_to_client_broadcast(broadcast_input)
    aggregation_input = daf.client_work(client_data, broadcast_output)
    aggregation_output = daf.client_to_server_aggregation(
        temp_server_state, aggregation_input
    )
    updated_server_state, server_output = daf.server_result(
        temp_server_state, aggregation_output
    )
    return updated_server_state, server_output

  return computation


def _check_type_is_fn(
    target: computation_types.Type,
    name: str,
@@ -1006,3 +1041,307 @@ def get_map_reduce_form_for_computation(
      for bb in blocks
  )
  return forms.MapReduceForm(comp.type_signature, *comps)


def get_distribute_aggregate_form_for_computation(
    comp: computation_impl.ConcreteComputation,
    *,
    tff_internal_preprocessing: Optional[BuildingBlockFn] = None,
) -> forms.DistributeAggregateForm:
  """Constructs `DistributeAggregateForm` for a computation.
  Args:
    comp: An instance of `computation_impl.ConcreteComputation` that is
      compatible with `DistributeAggregateForm`. The computation must take
      exactly two arguments, and the first must be a state value placed at
      `SERVER`. The computation must return exactly two values. The type of the
      first element in the result must also be assignable to the first element
      of the parameter.
    tff_internal_preprocessing: An optional function to transform the AST of the
      iterative process.
  Returns:
    An instance of `tff.backends.mapreduce.DistributeAggregateForm` equivalent
    to the provided `computation_base.Computation`.
  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  py_typecheck.check_type(comp, computation_base.Computation)

  # Apply any requested preprocessing to the computation.
  comp_tree = comp.to_building_block()
  if tff_internal_preprocessing is not None:
    comp_tree = tff_internal_preprocessing(comp_tree)

  # Check that the computation has the expected structure.
  comp_type = comp_tree.type_signature
  _check_type_is_fn(comp_type, '`comp`', TypeError)
  if not comp_type.parameter.is_struct() or len(comp_type.parameter) != 2:
    raise TypeError(
        'Expected `comp` to take two arguments, found parameter '
        f' type:\n{comp_type.parameter}'
    )
  if not comp_type.result.is_struct() or len(comp_type.result) != 2:
    raise TypeError(
        'Expected `comp` to return two values, found result '
        f'type:\n{comp_type.result}'
    )
  comp_tree = _replace_lambda_body_with_call_dominant_form(comp_tree)
  comp_tree, _ = tree_transformations.uniquify_reference_names(comp_tree)
  tree_analysis.check_broadcast_not_dependent_on_aggregate(comp_tree)

  # To generate the DistributeAggregateForm for the computation, we will split
  # the computation twice, first on broadcast intrinsics and then on aggregation
  # intrinsics. To ensure that any non-client-placed unbound refs used by the
  # aggregation intrinsics args are fed via the temporary state (as opposed to
  # via the clients, which shouldn't be returning non-client-placed data), add a
  # special broadcast call to the computation that depends on these args (if any
  # exist). Examples of these args are the `modulus` for a
  # federated_secure_modular_sum call or the `zero` for a federated_aggregate
  # call. Note that this injected broadcast call will not actually broadcast
  # these args to the clients (it broadcasts an empty struct). The sole purpose
  # of the injected broadcast is to establish a dependency that forces the calls
  # associated with these non-client-placed refs to appear in the *first* part
  # of the split on broadcast intrinsics rather than potentially appearing in
  # the *last* part of the split on broadcast intrinsics.
  args_needing_broadcast_dependency = []
  unbound_refs = transformation_utils.get_map_of_unbound_references(comp_tree)

  def _find_non_client_placed_args(inner_comp):
    # Examine the args of the aggregation intrinsic calls.
    if (
        inner_comp.is_call()
        and inner_comp.function.is_intrinsic()
        and inner_comp.function.intrinsic_def().aggregation_kind
    ):
      aggregation_args = (
          inner_comp.argument
          if inner_comp.argument.is_struct()
          else [inner_comp.argument]
      )
      unbound_ref_names_for_intrinsic = unbound_refs[inner_comp.argument]

      for aggregation_arg in aggregation_args:
        if unbound_refs[aggregation_arg].issubset(
            unbound_ref_names_for_intrinsic
        ):
          # If the arg is non-placed or server-placed, prepare to create a
          # federated broadcast that depends on it by normalizing it to a
          # server-placed value.
          if not aggregation_arg.type_signature.is_federated():
            has_placement_predicate = lambda x: x.type_signature.is_federated()
            if (
                tree_analysis.count(aggregation_arg, has_placement_predicate)
                > 0
            ):
              raise TypeError(
                  'DistributeAggregateForm cannot handle an aggregation '
                  f'intrinsic arg with type {aggregation_arg.type_signature}'
              )
            args_needing_broadcast_dependency.append(
                building_block_factory.create_federated_value(
                    aggregation_arg, placements.SERVER
                )
            )
          elif aggregation_arg.type_signature.placement == placements.SERVER:
            args_needing_broadcast_dependency.append(aggregation_arg)

      return inner_comp, True
    return inner_comp, False

  tree_analysis.visit_preorder(comp_tree, _find_non_client_placed_args)

  # Add an injected broadcast call to the computation that depends on the
  # identified non-client-placed args, if any exist. To avoid broadcasting the
  # actual non-client-placed args (undesirable from both a privacy and
  # efficiency standpoint), instead broadcast the empty struct result generated
  # by an intermediate map call that takes the non-client-placed args as input.
  # This approach should work as long as the intermediate map call does not get
  # pruned by various tree transformations. Currently, tree transformations
  # such as to_call_dominant do not recognize that our intermediate map call
  # here can be drastically simplified.
  if args_needing_broadcast_dependency:
    zipped_args_needing_broadcast_dependency = (
        building_block_factory.create_federated_zip(
            building_blocks.Struct(args_needing_broadcast_dependency)
        )
    )
    injected_broadcast = building_block_factory.create_federated_broadcast(
        building_block_factory.create_federated_apply(
            building_blocks.Lambda(
                'ignored_param',
                zipped_args_needing_broadcast_dependency.type_signature.member,
                building_blocks.Struct([]),
            ),
            zipped_args_needing_broadcast_dependency,
        )
    )
    # Add the injected broadcast call to the block locals.
    revised_block_locals = comp_tree.result.locals + [(
        'injected_broadcast_ref',
        injected_broadcast,
    )]
    # Add a reference to the injected broadcast call in the result so that it
    # does not get pruned by various tree transformations. We will remove this
    # additional element in the result after the first split operation.
    revised_block_result = structure.to_elements(comp_tree.result.result) + [
        building_blocks.Reference(
            'injected_broadcast_ref',
            injected_broadcast.type_signature,
        )
    ]
    comp_tree = building_blocks.Lambda(
        comp_tree.parameter_name,
        comp_tree.parameter_type,
        building_blocks.Block(
            revised_block_locals,
            building_blocks.Struct(revised_block_result),
        ),
    )

  # Split first on the broadcast intrinsics.
  # - The "before" comp in this split (which will eventually become the
  # server_prepare portion of the DAF) should only depend on the server portion
  # of the original comp input.
  # - The "intrinsic" comp in this split (which will eventually become the
  # server_to_client_broadcast portion of the DAF) should not depend on any
  # portion of the original comp.
  # - The "after" comp in this split (which will eventually become the
  # client_work, client_to_server_aggregation, and server_result portions of
  # the DAF) should be allowed to depend only the client portion of the original
  # comp. Any server-related or non-placed dependencies will be passed via the
  # intermediate state.
  server_state_index = 0
  client_data_index = 1
  server_prepare, server_to_client_broadcast, after_broadcast = (
      transformations.divisive_force_align_and_split_by_intrinsics(
          comp_tree,
          intrinsic_defs.get_broadcast_intrinsics(),
          before_comp_allowed_original_arg_subparameters=[
              (server_state_index,)
          ],
          intrinsic_comp_allowed_original_arg_subparameters=[],
          after_comp_allowed_original_arg_subparameters=[(client_data_index,)],
      )
  )

  # Helper method to replace a lambda with parameter that is a single-element
  # struct with a lambda that uses the element directly.
  def _unnest_lambda_parameter(comp):
    assert comp.is_lambda()
    assert comp.parameter_type.is_struct()

    name_generator = building_block_factory.unique_name_generator(comp)
    new_param_name = next(name_generator)
    replacement_ref = building_blocks.Reference(
        new_param_name, comp.parameter_type[0]
    )
    modified_comp_body = tree_transformations.replace_selections(
        comp.result, comp.parameter_name, {(0,): replacement_ref}
    )
    modified_comp = building_blocks.Lambda(
        replacement_ref.name, replacement_ref.type_signature, modified_comp_body
    )
    tree_analysis.check_contains_no_unbound_references(modified_comp)

    return modified_comp

  # Finalize the server_prepare and server_to_client_broadcast comps by
  # removing a layer of nesting from the input parameter.
  server_prepare = _unnest_lambda_parameter(server_prepare)
  server_to_client_broadcast = _unnest_lambda_parameter(
      server_to_client_broadcast
  )

  # Next split on the aggregation intrinsics. If an injected broadcast was used
  # in the previous step, drop the last element in the output (which held the
  # output of the injected broadcast) before performing the second split.
  # - The "before" comp in this split (which will eventually become the
  # client_work portion of the DAF) should only depend on the client portion of
  # the original comp input (i.e. the client portion of the "original_arg"
  # portion of the after_broadcast input) and the portion of the after_broadcast
  # input that represents the intrinsic output that was produced in the first
  # split (i.e. the broadcast output).
  # - The "intrinsic" comp in this split (which will eventually become the
  # client_to_server_aggregation portion of the DAF) should only depend on the
  # portion of the after_broadcast input that represents the intermediate state
  # that was produced in the first split.
  # - The "after" comp in this split (which will eventually become the
  # server_result portion of the DAF) should only depend on the server portion
  # of the original comp input (i.e. the server portion of the "original_arg"
  # portion of the after_broadcast input) and the portion of the after_broadcast
  # input that represents the intermediate state that was produced in the first
  # split.
  if args_needing_broadcast_dependency:
    assert after_broadcast.result.result.is_struct()
    # Check that the last element of the result is the expected empty struct
    # associated with the injected broadcast call.
    result_len = len(after_broadcast.result.result)
    injected_broadcast_result = after_broadcast.result.result[result_len - 1]
    assert injected_broadcast_result.type_signature.member.is_struct()
    assert not injected_broadcast_result.type_signature.member
    after_broadcast = building_blocks.Lambda(
        after_broadcast.parameter_name,
        after_broadcast.parameter_type,
        building_blocks.Block(
            after_broadcast.result.locals,
            building_blocks.Struct(
                structure.to_elements(after_broadcast.result.result)[:-1]
            ),
        ),
    )
  client_data_index_in_after_broadcast_param = 0
  intrinsic_results_index_in_after_broadcast_param = 1
  intermediate_state_index_in_after_broadcast_param = 2
  client_work, client_to_server_aggregation, server_result = (
      transformations.divisive_force_align_and_split_by_intrinsics(
          after_broadcast,
          intrinsic_defs.get_aggregation_intrinsics(),
          before_comp_allowed_original_arg_subparameters=[
              (client_data_index_in_after_broadcast_param,),
              (intrinsic_results_index_in_after_broadcast_param,),
          ],
          intrinsic_comp_allowed_original_arg_subparameters=[
              (intermediate_state_index_in_after_broadcast_param,)
          ],
          after_comp_allowed_original_arg_subparameters=[
              (intermediate_state_index_in_after_broadcast_param,),
          ],
      )
  )

  # Drop the intermediate_state produced by the second split that is part of
  # the client_work output.
  index_of_intrinsic_args_in_client_work_result = 0
  client_work = building_block_factory.select_output_from_lambda(
      client_work, index_of_intrinsic_args_in_client_work_result
  )

  # Drop the intermediate_state param produced by the second split that is part
  # of the server_result parameter (but keep the part of the param that
  # corresponds to the intermediate_state produced by the first split).
  intermediate_state_index_in_server_result_param = 0
  aggregation_result_index_in_server_result_param = 1
  server_result = tree_transformations.as_function_of_some_subparameters(
      server_result,
      [
          (intermediate_state_index_in_server_result_param,),
          (aggregation_result_index_in_server_result_param,),
      ],
  )

  blocks = (
      server_prepare,
      server_to_client_broadcast,
      client_work,
      client_to_server_aggregation,
      server_result,
  )

  comps = (
      computation_impl.ConcreteComputation.from_building_block(bb)
      for bb in blocks
  )

  return forms.DistributeAggregateForm(comp.type_signature, *comps)
  190  
tensorflow_federated/python/core/backends/mapreduce/form_utils_test.py
@@ -18,6 +18,7 @@
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.mapreduce import distribute_aggregate_test_utils
from tensorflow_federated.python.core.backends.mapreduce import form_utils
from tensorflow_federated.python.core.backends.mapreduce import forms
from tensorflow_federated.python.core.backends.mapreduce import mapreduce_test_utils
@@ -540,7 +541,7 @@ def get_example_cf_compatible_iterative_processes():
  # pyformat: enable


class MapReduceFormTestCase(tf.test.TestCase):
class FederatedFormTestCase(tf.test.TestCase):
  """A base class that overrides evaluate to handle various executors."""

  def evaluate(self, value):
@@ -554,8 +555,177 @@ def evaluate(self, value):
      )


class GetComputationForDistributeAggregateFormTest(
    FederatedFormTestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      (
          'temperature',
          distribute_aggregate_test_utils.get_temperature_sensor_example(),
      ),
      ('mnist', distribute_aggregate_test_utils.get_mnist_training_example()),
  )
  def test_type_signature_matches_generated_computation(self, example):
    comp = form_utils.get_computation_for_distribute_aggregate_form(example.daf)
    self.assertTrue(
        comp.type_signature.is_equivalent_to(example.daf.type_signature)
    )

  def test_with_temperature_sensor_example(self):
    example = distribute_aggregate_test_utils.get_temperature_sensor_example()

    state = example.initialize()

    comp = form_utils.get_computation_for_distribute_aggregate_form(example.daf)
    state, metrics = comp(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertAllEqual(state, collections.OrderedDict(num_rounds=1))
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.5)
    )

    state, metrics = comp(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.75)
    )


class GetDistributeAggregateFormTest(
    FederatedFormTestCase, parameterized.TestCase
):

  def test_next_computation_returning_tensor_fails_well(self):
    initialize = (
        distribute_aggregate_test_utils.get_temperature_sensor_example().initialize
    )
    init_result = initialize.type_signature.result
    lam = building_blocks.Lambda(
        'x', init_result, building_blocks.Reference('x', init_result)
    )
    bad_comp = computation_impl.ConcreteComputation.from_building_block(lam)
    with self.assertRaises(TypeError):
      form_utils.get_distribute_aggregate_form_for_computation(bad_comp)

  def test_broadcast_dependent_on_aggregate_fails_well(self):
    example = distribute_aggregate_test_utils.get_mnist_training_example()
    comp = form_utils.get_computation_for_distribute_aggregate_form(example.daf)
    comp_bb = comp.to_building_block()
    top_level_param = building_blocks.Reference(
        comp_bb.parameter_name, comp_bb.parameter_type
    )
    first_result = building_blocks.Call(comp_bb, top_level_param)
    middle_param = building_blocks.Struct([
        building_blocks.Selection(first_result, index=0),
        building_blocks.Selection(top_level_param, index=1),
    ])
    second_result = building_blocks.Call(comp_bb, middle_param)
    not_reducible = building_blocks.Lambda(
        comp_bb.parameter_name, comp_bb.parameter_type, second_result
    )
    bad_comp = computation_impl.ConcreteComputation.from_building_block(
        not_reducible
    )
    with self.assertRaisesRegex(ValueError, 'broadcast dependent on aggregate'):
      form_utils.get_distribute_aggregate_form_for_computation(bad_comp)

  def test_gets_distribute_aggregate_form_for_nested_broadcast(self):
    comp = get_computation_with_nested_broadcasts()
    daf = form_utils.get_distribute_aggregate_form_for_computation(comp)
    self.assertIsInstance(daf, forms.DistributeAggregateForm)

  def test_constructs_distribute_aggregate_form_from_mnist_training_example(
      self,
  ):
    comp = form_utils.get_computation_for_distribute_aggregate_form(
        distribute_aggregate_test_utils.get_mnist_training_example().daf
    )
    daf = form_utils.get_distribute_aggregate_form_for_computation(comp)
    self.assertIsInstance(daf, forms.DistributeAggregateForm)

  def test_temperature_example_round_trip(self):
    example = distribute_aggregate_test_utils.get_temperature_sensor_example()
    comp = form_utils.get_computation_for_distribute_aggregate_form(example.daf)
    new_initialize = form_utils.get_state_initialization_computation(
        example.initialize
    )
    new_daf = form_utils.get_distribute_aggregate_form_for_computation(comp)
    new_comp = form_utils.get_computation_for_distribute_aggregate_form(new_daf)

    state = new_initialize()
    self.assertEqual(state['num_rounds'], 0)

    state, metrics = new_comp(state, [[28.0], [30.0, 33.0, 29.0]])
    self.assertEqual(state['num_rounds'], 1)
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.5)
    )

    state, metrics = new_comp(state, [[33.0], [34.0], [35.0], [36.0]])
    self.assertAllClose(
        metrics, collections.OrderedDict(ratio_over_threshold=0.75)
    )
    # Check that no TF work has been unintentionally duplicated.
    self.assertEqual(
        tree_analysis.count_tensorflow_variables_under(
            comp.to_building_block()
        ),
        tree_analysis.count_tensorflow_variables_under(
            new_comp.to_building_block()
        ),
    )

  def test_mnist_training_round_trip(self):
    example = distribute_aggregate_test_utils.get_mnist_training_example()
    comp = form_utils.get_computation_for_distribute_aggregate_form(example.daf)
    new_initialize = form_utils.get_state_initialization_computation(
        example.initialize
    )
    new_daf = form_utils.get_distribute_aggregate_form_for_computation(comp)
    new_comp = form_utils.get_computation_for_distribute_aggregate_form(new_daf)

    state1 = example.initialize()
    state2 = new_initialize()
    self.assertAllClose(state1, state2)
    whimsy_x = np.array([[0.5] * 784], dtype=np.float32)
    whimsy_y = np.array([1], dtype=np.int32)
    client_data = [collections.OrderedDict(x=whimsy_x, y=whimsy_y)]
    round_1 = new_comp(state1, [client_data])
    state = round_1[0]
    metrics = round_1[1]
    alt_round_1 = new_comp(state2, [client_data])
    alt_state = alt_round_1[0]
    self.assertAllClose(state, alt_state)
    alt_metrics = alt_round_1[1]
    self.assertAllClose(metrics, alt_metrics)
    # Check that no TF work has been unintentionally duplicated.
    self.assertEqual(
        tree_analysis.count_tensorflow_variables_under(
            comp.to_building_block()
        ),
        tree_analysis.count_tensorflow_variables_under(
            new_comp.to_building_block()
        ),
    )

  @parameterized.named_parameters(
      *get_example_cf_compatible_iterative_processes()
  )
  def test_returns_distribute_aggregate_form(self, ip):
    daf = form_utils.get_distribute_aggregate_form_for_computation(ip.next)
    self.assertIsInstance(daf, forms.DistributeAggregateForm)

  def test_returns_distribute_aggregate_form_with_indirection_to_intrinsic(
      self,
  ):
    ip = (
        mapreduce_test_utils.get_iterative_process_for_example_with_lambda_returning_aggregation()
    )
    daf = form_utils.get_distribute_aggregate_form_for_computation(ip.next)
    self.assertIsInstance(daf, forms.DistributeAggregateForm)


class GetComputationForMapReduceFormTest(
    MapReduceFormTestCase, parameterized.TestCase
    FederatedFormTestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
@@ -587,7 +757,7 @@ def test_with_temperature_sensor_example(self):


class CheckMapReduceFormCompatibleWithComputationTest(
    MapReduceFormTestCase, parameterized.TestCase
    FederatedFormTestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
@@ -616,7 +786,7 @@ def comp(server_state, client_data):
      form_utils.check_computation_compatible_with_map_reduce_form(comp)


class GetMapReduceFormTest(MapReduceFormTestCase, parameterized.TestCase):
class GetMapReduceFormTest(FederatedFormTestCase, parameterized.TestCase):

  def test_next_computation_returning_tensor_fails_well(self):
    initialize = (
@@ -670,10 +840,8 @@ def test_temperature_example_round_trip(self):
    # to lose the python container annotations on the StructType.
    example = mapreduce_test_utils.get_temperature_sensor_example()
    comp = form_utils.get_computation_for_map_reduce_form(example.mrf)
    new_initialize = (
        form_utils.get_state_initialization_computation_for_map_reduce_form(
            example.initialize
        )
    new_initialize = form_utils.get_state_initialization_computation(
        example.initialize
    )
    new_mrf = form_utils.get_map_reduce_form_for_computation(comp)
    new_comp = form_utils.get_computation_for_map_reduce_form(new_mrf)
@@ -708,10 +876,8 @@ def test_mnist_training_round_trip(self):
    # execution is C++-backed, this can go away.
    grappler_config = tf.compat.v1.ConfigProto()
    grappler_config.graph_options.rewrite_options.disable_meta_optimizer = True
    new_initialize = (
        form_utils.get_state_initialization_computation_for_map_reduce_form(
            example.initialize
        )
    new_initialize = form_utils.get_state_initialization_computation(
        example.initialize
    )
    new_mrf = form_utils.get_map_reduce_form_for_computation(
        comp, grappler_config
  24  
tensorflow_federated/python/core/backends/mapreduce/forms_test.py
@@ -1014,6 +1014,30 @@ def test_init_raises_type_error_with_type_signature_mismatch(self):
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
  20  
tensorflow_federated/python/core/impl/compiler/intrinsic_defs.py
@@ -659,3 +659,23 @@ def __repr__(self):

def uri_to_intrinsic_def(uri) -> Optional[IntrinsicDef]:
  return _intrinsic_registry.get(uri)


# TODO(b/254770431): Add documentation explaining the implications of setting
# broadcast_kind for an intrinsic.
def get_broadcast_intrinsics() -> list[IntrinsicDef]:
  return [
      intrinsic
      for intrinsic in _intrinsic_registry.values()
      if intrinsic.broadcast_kind
  ]


# TODO(b/254770431): Add documentation explaining the implications of setting
# aggregation_kind for an intrinsic.
def get_aggregation_intrinsics() -> list[IntrinsicDef]:
  return [
      intrinsic
      for intrinsic in _intrinsic_registry.values()
      if intrinsic.aggregation_kind

package(default_visibility = ["//visibility:open"])

licenses(["notice"])

exports_files(["LICENSE"])
