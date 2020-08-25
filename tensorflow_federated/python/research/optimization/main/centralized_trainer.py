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
"""Runs centralized training on various tasks with different optimizers.

The tasks, optimizers, and hyperparameters are all governed via flags. For more
details on the supported optimizers, see `shared/optimizer_utils.py`. For more
details on how the training loop is performed, see
`shared/centralized_training_loop.py`.
"""

import collections

from absl import app
from absl import flags

from tensorflow_federated.python.research.optimization.cifar100 import centralized_cifar100
from tensorflow_federated.python.research.optimization.emnist import centralized_emnist
from tensorflow_federated.python.research.optimization.emnist_ae import centralized_emnist_ae
from tensorflow_federated.python.research.optimization.shakespeare import centralized_shakespeare
from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.optimization.stackoverflow import centralized_stackoverflow
from tensorflow_federated.python.research.optimization.stackoverflow_lr import centralized_stackoverflow_lr
from tensorflow_federated.python.research.utils import utils_impl

_SUPPORTED_TASKS = [
    'cifar100', 'emnist_cr', 'emnist_ae', 'shakespeare', 'stackoverflow_nwp',
    'stackoverflow_lr'
]

with utils_impl.record_new_flags() as hparam_flags:
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

  # Generic centralized training flags
  optimizer_utils.define_optimizer_flags('centralized')
  flags.DEFINE_string(
      'experiment_name', None,
      'Name of the experiment. Part of the name of the output directory.')
  flags.DEFINE_string(
      'root_output_dir', '/tmp/centralized_opt',
      'The top-level output directory experiment runs. --experiment_name will '
      'be appended, and the directory will contain tensorboard logs, metrics '
      'written as CSVs, and a CSV of hyperparameter choices.')
  flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train.')
  flags.DEFINE_integer('batch_size', 32,
                       'Size of batches for training and eval.')
  flags.DEFINE_integer('decay_epochs', None, 'Number of epochs before decaying '
                       'the learning rate.')
  flags.DEFINE_float('lr_decay', None, 'How much to decay the learning rate by'
                     ' at each stage.')

  # CIFAR-100 flags
  flags.DEFINE_integer('cifar100_crop_size', 24, 'The height and width of '
                       'images after preprocessing.')

  # EMNIST character recognition flags
  flags.DEFINE_enum('emnist_cr_model', 'cnn', ['cnn', '2nn'],
                    'Which model to use for classification.')

  # Shakespeare next character prediction flags
  flags.DEFINE_integer(
      'shakespeare_sequence_length', 80,
      'Length of character sequences to use for the RNN model.')

  # Stack Overflow NWP flags
  flags.DEFINE_integer('so_nwp_vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('so_nwp_num_oov_buckets', 1,
                       'Number of out of vocabulary buckets.')
  flags.DEFINE_integer('so_nwp_sequence_length', 20,
                       'Max sequence length to use.')
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

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  optimizer = optimizer_utils.create_optimizer_fn_from_flags('centralized')()
  hparams_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])

  common_args = collections.OrderedDict([
      ('optimizer', optimizer),
      ('experiment_name', FLAGS.experiment_name),
      ('root_output_dir', FLAGS.root_output_dir),
      ('num_epochs', FLAGS.num_epochs),
      ('batch_size', FLAGS.batch_size),
      ('decay_epochs', FLAGS.decay_epochs),
      ('lr_decay', FLAGS.lr_decay),
      ('hparams_dict', hparams_dict),
  ])

  if FLAGS.task == 'cifar100':
    centralized_cifar100.run_centralized(
        **common_args, crop_size=FLAGS.cifar100_crop_size)

  elif FLAGS.task == 'emnist_cr':
    centralized_emnist.run_centralized(
        **common_args, emnist_model=FLAGS.emnist_cr_model)

  elif FLAGS.task == 'emnist_ae':
    centralized_emnist_ae.run_centralized(**common_args)

  elif FLAGS.task == 'shakespeare':
    centralized_shakespeare.run_centralized(
        **common_args, sequence_length=FLAGS.shakespeare_sequence_length)

  elif FLAGS.task == 'stackoverflow_nwp':
    so_nwp_flags = collections.OrderedDict()
    for flag_name in FLAGS:
      if flag_name.startswith('so_nwp_'):
        so_nwp_flags[flag_name[7:]] = FLAGS[flag_name].value
    centralized_stackoverflow.run_centralized(**common_args, **so_nwp_flags)

  elif FLAGS.task == 'stackoverflow_lr':
    so_lr_flags = collections.OrderedDict()
    for flag_name in FLAGS:
      if flag_name.startswith('so_lr_'):
        so_lr_flags[flag_name[6:]] = FLAGS[flag_name].value
    centralized_stackoverflow_lr.run_centralized(**common_args, **so_lr_flags)

  else:
    raise ValueError(
        '--task flag {} is not supported, must be one of {}.'.format(
            FLAGS.task, _SUPPORTED_TASKS))


if __name__ == '__main__':
  app.run(main)
