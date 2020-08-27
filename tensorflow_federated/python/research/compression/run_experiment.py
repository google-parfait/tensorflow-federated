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
"""An example training loop lossily compressing the server/client communication.

Example command line flags to use to run an experiment:
--client_optimizer=sgd
--client_learning_rate=0.2
--server_optimizer=sgd
--server_learning_rate=1.0
--use_compression=True
--broadcast_quantization_bits=8
--aggregation_quantization_bits=8
--use_sparsity_in_aggregation=True
"""

import functools

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.compression import sparsity
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils import utils_impl
from tensorflow_federated.python.research.utils.datasets import emnist_dataset
from tensorflow_federated.python.research.utils.models import emnist_models
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


with utils_impl.record_new_flags():
  # Training hyperparameters
  flags.DEFINE_integer('clients_per_round', 2,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20,
                       'Batch size used on the client.')
  flags.DEFINE_boolean(
      'only_digits', True, 'Whether to use the digit-only '
      'EMNIST dataset (10 characters) or the extended EMNIST '
      'dataset (62 characters).')

  # Optimizer configuration (this defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('server')
  utils_impl.define_optimizer_flags('client')

  # Compression hyperparameters.
  flags.DEFINE_boolean('use_compression', True,
                       'Whether to use compression code path.')
  flags.DEFINE_integer(
      'broadcast_quantization_bits', 8,
      'Number of quantization bits for server to client '
      'compression.')
  flags.DEFINE_integer(
      'aggregation_quantization_bits', 8,
      'Number of quantization bits for client to server '
      'compression.')
  flags.DEFINE_boolean('use_sparsity_in_aggregation', True,
                       'Whether to add sparsity to the aggregation. This will '
                       'only be used for client to server compression.')

with utils_impl.record_new_flags() as training_loop_flags:
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/compression/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_boolean(
      'write_metrics_with_bz2', True, 'Whether to use bz2 '
      'compression when writing output metrics to a csv file.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')
  flags.DEFINE_integer(
      'rounds_per_profile', 0,
      '(Experimental) How often to run the experimental TF profiler, if >0.')

# End of hyperparameter flags.

FLAGS = flags.FLAGS


def assign_weights_fn(reference_model, keras_model):
  """Assign tff.learning.ModelWeights to the weights of a `tf.keras.Model`.

  Args:
    reference_model: the `tff.learning.ModelWeights` object to assign weights
      from.
    keras_model: the `tf.keras.Model` object to assign weights to.
  """
  if not isinstance(reference_model, tff.learning.ModelWeights):
    raise TypeError('The reference model must be an instance of '
                    'tff.learning.ModelWeights.')

  def assign_weights(keras_weights, tff_weights):
    for k, w in zip(keras_weights, tff_weights):
      k.assign(w)

  assign_weights(keras_model.trainable_weights, reference_model.trainable)
  assign_weights(keras_model.non_trainable_weights,
                 reference_model.non_trainable)


def model_builder():
  """Create a keras model based on the original FedAvg CNN."""
  return emnist_models.create_original_fedavg_cnn_model(
      only_digits=FLAGS.only_digits)


def loss_builder():
  return tf.keras.losses.SparseCategoricalCrossentropy()


def metrics_builder():
  return [tf.keras.metrics.SparseCategoricalAccuracy()]


def _broadcast_encoder_fn(value):
  """Function for building encoded broadcast.

  This method decides, based on the tensor size, whether to use lossy
  compression or keep it as is (use identity encoder). The motivation for this
  pattern is due to the fact that compression of small model weights can provide
  only negligible benefit, while at the same time, lossy compression of small
  weights usually results in larger impact on model's accuracy.

  Args:
    value: A tensor or variable to be encoded in server to client communication.

  Returns:
    A `te.core.SimpleEncoder`.
  """
  # TODO(b/131681951): We cannot use .from_tensor(...) because it does not
  # currently support Variables.
  spec = tf.TensorSpec(value.shape, value.dtype)
  if value.shape.num_elements() > 10000:
    return te.encoders.as_simple_encoder(
        te.encoders.uniform_quantization(FLAGS.broadcast_quantization_bits),
        spec)
  else:
    return te.encoders.as_simple_encoder(te.encoders.identity(), spec)


def _mean_encoder_fn(value):
  """Function for building encoded mean.

  This method decides, based on the tensor size, whether to use lossy
  compression or keep it as is (use identity encoder). The motivation for this
  pattern is due to the fact that compression of small model weights can provide
  only negligible benefit, while at the same time, lossy compression of small
  weights usually results in larger impact on model's accuracy.

  Args:
    value: A tensor or variable to be encoded in client to server communication.

  Returns:
    A `te.core.GatherEncoder`.
  """
  # TODO(b/131681951): We cannot use .from_tensor(...) because it does not
  # currently support Variables.
  spec = tf.TensorSpec(value.shape, value.dtype)
  if value.shape.num_elements() > 10000:
    if FLAGS.use_sparsity_in_aggregation:
      return te.encoders.as_gather_encoder(
          sparsity.sparse_quantizing_encoder(
              FLAGS.aggregation_quantization_bits), spec)
    else:
      return te.encoders.as_gather_encoder(
          te.encoders.uniform_quantization(FLAGS.aggregation_quantization_bits),
          spec)
  else:
    return te.encoders.as_gather_encoder(te.encoders.identity(), spec)


def run_experiment():
  """Data preprocessing and experiment execution."""
  emnist_train, emnist_test = emnist_dataset.get_emnist_datasets(
      FLAGS.client_batch_size,
      FLAGS.client_epochs_per_round,
      only_digits=FLAGS.only_digits)

  example_dataset = emnist_train.create_tf_dataset_for_client(
      emnist_train.client_ids[0])
  input_spec = example_dataset.element_spec

  client_datasets_fn = training_utils.build_client_datasets_fn(
      emnist_train, FLAGS.clients_per_round)

  evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=emnist_test,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  client_optimizer_fn = functools.partial(
      utils_impl.create_optimizer_from_flags, 'client')
  server_optimizer_fn = functools.partial(
      utils_impl.create_optimizer_from_flags, 'server')

  def tff_model_fn():
    keras_model = model_builder()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  if FLAGS.use_compression:
    # We create a `MeasuredProcess` for broadcast process and a
    # `MeasuredProcess` for aggregate process by providing the
    # `_broadcast_encoder_fn` and `_mean_encoder_fn` to corresponding utilities.
    # The fns are called once for each of the model weights created by
    # tff_model_fn, and return instances of appropriate encoders.
    encoded_broadcast_process = (
        tff.learning.framework.build_encoded_broadcast_process_from_model(
            tff_model_fn, _broadcast_encoder_fn))
    encoded_mean_process = (
        tff.learning.framework.build_encoded_mean_process_from_model(
            tff_model_fn, _mean_encoder_fn))
  else:
    encoded_broadcast_process = None
    encoded_mean_process = None

  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn=tff_model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn,
      aggregation_process=encoded_mean_process,
      broadcast_process=encoded_broadcast_process)

  hparam_dict = utils_impl.lookup_flag_values(utils_impl.get_hparam_flags())
  training_loop_dict = utils_impl.lookup_flag_values(training_loop_flags)

  training_loop.run(
      iterative_process=iterative_process,
      client_datasets_fn=client_datasets_fn,
      validation_fn=evaluate_fn,
      hparam_dict=hparam_dict,
      **training_loop_dict)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))
  tff.backends.native.set_local_execution_context(max_fanout=25)
  run_experiment()


if __name__ == '__main__':
  app.run(main)
