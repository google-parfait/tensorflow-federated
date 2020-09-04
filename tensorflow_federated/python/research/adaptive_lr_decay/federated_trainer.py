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
"""Runs federated training on various tasks using a generalized form of FedAvg.

Specifically, we create (according to flags) an iterative processes that adapts
the client and server learning rate according to the history of loss values
encountered throughout training. For more details on the learning rate decay,
see `callbacks.py` and `adaptive_fed_avg.py`.
"""

import collections
from typing import Any, Callable, Optional

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.adaptive_lr_decay import adaptive_fed_avg
from tensorflow_federated.python.research.adaptive_lr_decay import callbacks
from tensorflow_federated.python.research.optimization.cifar100 import federated_cifar100
from tensorflow_federated.python.research.optimization.emnist import federated_emnist
from tensorflow_federated.python.research.optimization.emnist_ae import federated_emnist_ae
from tensorflow_federated.python.research.optimization.shakespeare import federated_shakespeare
from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.optimization.stackoverflow import federated_stackoverflow
from tensorflow_federated.python.research.optimization.stackoverflow_lr import federated_stackoverflow_lr
from tensorflow_federated.python.research.utils import utils_impl

_SUPPORTED_TASKS = [
    'cifar100', 'emnist_cr', 'emnist_ae', 'shakespeare', 'stackoverflow_nwp',
    'stackoverflow_lr'
]

with utils_impl.record_hparam_flags() as optimizer_flags:
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as callback_flags:
  flags.DEFINE_float(
      'client_decay_factor', 0.1, 'Amount to decay the client learning rate '
      'upon reaching a plateau.')
  flags.DEFINE_float(
      'server_decay_factor', 0.9, 'Amount to decay the server learning rate '
      'upon reaching a plateau.')
  flags.DEFINE_float(
      'min_delta', 1e-4, 'Minimum delta for improvement in the learning rate '
      'callbacks.')
  flags.DEFINE_integer(
      'window_size', 100, 'Number of rounds to take a moving average over when '
      'estimating the training loss in learning rate callbacks.')
  flags.DEFINE_integer(
      'patience', 100, 'Number of rounds of non-improvement before decaying the'
      'learning rate.')
  flags.DEFINE_float('min_lr', 0.0, 'The minimum learning rate.')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer(
      'client_epochs_per_round', -1,
      'Number of epochs in the client to take per round. If '
      'set to -1, the dataset repeats indefinitely, unless '
      'max_batches_per_client is set to some positive value.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer(
      'max_batches_per_client', 10, 'Maximum number of train '
      'steps each client performs per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')

  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/adaptive_lr_decay/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_boolean(
      'write_metrics_with_bz2', True, 'Whether to use bz2 '
      'compression when writing output metrics to a csv file.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer(
      'rounds_per_train_eval', 100,
      'How often to evaluate the global model on the entire training dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')
  flags.DEFINE_integer(
      'rounds_per_profile', 0,
      '(Experimental) How often to run the experimental TF profiler, if >0.')

with utils_impl.record_hparam_flags() as task_flags:
  # Task specification
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

  # CIFAR-100 flags
  flags.DEFINE_integer('cifar100_crop_size', 24, 'The height and width of '
                       'images after preprocessing.')

  # EMNIST CR flags
  flags.DEFINE_enum(
      'emnist_cr_model', 'cnn', ['cnn', '2nn'], 'Which model to '
      'use. This can be a convolutional model (cnn) or a two '
      'hidden-layer densely connected network (2nn).')

  # Shakespeare flags
  flags.DEFINE_integer(
      'shakespeare_sequence_length', 80,
      'Length of character sequences to use for the RNN model.')

  # Stack Overflow NWP flags
  flags.DEFINE_integer('so_nwp_vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('so_nwp_num_oov_buckets', 1,
                       'Number of out of vocabulary buckets.')
  flags.DEFINE_integer('so_nwp_sequence_length', 20,
                       'Max sequence length to use.')
  flags.DEFINE_integer('so_nwp_max_elements_per_user', 1000, 'Max number of '
                       'training sentences to use per user.')
  flags.DEFINE_integer(
      'so_nwp_num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')
  flags.DEFINE_integer('so_nwp_embedding_size', 96,
                       'Dimension of word embedding to use.')
  flags.DEFINE_integer('so_nwp_latent_size', 670,
                       'Dimension of latent size to use in recurrent cell')
  flags.DEFINE_integer('so_nwp_num_layers', 1,
                       'Number of stacked recurrent layers to use.')
  flags.DEFINE_boolean(
      'so_nwp_shared_embedding', False,
      'Boolean indicating whether to tie input and output embeddings.')

  # Stack Overflow LR flags
  flags.DEFINE_integer('so_lr_vocab_tokens_size', 10000,
                       'Vocab tokens size used.')
  flags.DEFINE_integer('so_lr_vocab_tags_size', 500, 'Vocab tags size used.')
  flags.DEFINE_integer(
      'so_lr_num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')
  flags.DEFINE_integer('so_lr_max_elements_per_user', 1000,
                       'Max number of training '
                       'sentences to use per user.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  client_lr_callback = callbacks.create_reduce_lr_on_plateau(
      learning_rate=FLAGS.client_learning_rate,
      decay_factor=FLAGS.client_decay_factor,
      min_delta=FLAGS.min_delta,
      min_lr=FLAGS.min_lr,
      window_size=FLAGS.window_size,
      patience=FLAGS.patience)

  server_lr_callback = callbacks.create_reduce_lr_on_plateau(
      learning_rate=FLAGS.server_learning_rate,
      decay_factor=FLAGS.server_decay_factor,
      min_delta=FLAGS.min_delta,
      min_lr=FLAGS.min_lr,
      window_size=FLAGS.window_size,
      patience=FLAGS.patience)

  def iterative_process_builder(
      model_fn: Callable[[], tff.learning.Model],
      client_weight_fn: Optional[Callable[[Any], tf.Tensor]] = None,
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.
      client_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor providing the weight
        in the federated average of model deltas. If not provided, the default
        is the total number of examples processed on device.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    return adaptive_fed_avg.build_fed_avg_process(
        model_fn,
        client_lr_callback,
        server_lr_callback,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_weight_fn=client_weight_fn)

  assign_weights_fn = adaptive_fed_avg.ServerState.assign_weights_to_keras_model
  hparam_dict = utils_impl.lookup_flag_values(utils_impl.get_hparam_flags())

  shared_args = utils_impl.lookup_flag_values(shared_flags)
  shared_args['iterative_process_builder'] = iterative_process_builder
  shared_args['assign_weights_fn'] = assign_weights_fn

  if FLAGS.task == 'cifar100':
    hparam_dict['cifar100_crop_size'] = FLAGS.cifar100_crop_size
    federated_cifar100.run_federated(
        **shared_args,
        crop_size=FLAGS.cifar100_crop_size,
        hparam_dict=hparam_dict)

  elif FLAGS.task == 'emnist_cr':
    federated_emnist.run_federated(
        **shared_args,
        emnist_model=FLAGS.emnist_cr_model,
        hparam_dict=hparam_dict)

  elif FLAGS.task == 'emnist_ae':
    federated_emnist_ae.run_federated(**shared_args, hparam_dict=hparam_dict)

  elif FLAGS.task == 'shakespeare':
    federated_shakespeare.run_federated(
        **shared_args,
        sequence_length=FLAGS.shakespeare_sequence_length,
        hparam_dict=hparam_dict)

  elif FLAGS.task == 'stackoverflow_nwp':
    so_nwp_flags = collections.OrderedDict()
    for flag_name in task_flags:
      if flag_name.startswith('so_nwp_'):
        so_nwp_flags[flag_name[7:]] = FLAGS[flag_name].value
    federated_stackoverflow.run_federated(
        **shared_args, **so_nwp_flags, hparam_dict=hparam_dict)

  elif FLAGS.task == 'stackoverflow_lr':
    so_lr_flags = collections.OrderedDict()
    for flag_name in task_flags:
      if flag_name.startswith('so_lr_'):
        so_lr_flags[flag_name[6:]] = FLAGS[flag_name].value
    federated_stackoverflow_lr.run_federated(
        **shared_args, **so_lr_flags, hparam_dict=hparam_dict)


if __name__ == '__main__':
  app.run(main)
