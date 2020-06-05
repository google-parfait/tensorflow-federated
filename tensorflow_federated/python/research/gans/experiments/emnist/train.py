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
"""Train DP Federated GAN to synthesize images resembling EMNIST data."""

import collections
import functools
import os.path
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_gan as tfgan
import tensorflow_privacy
import tree

from tensorboard.plugins.hparams import api as hp
from tensorflow_federated.python.research.gans import gan_losses
from tensorflow_federated.python.research.gans import gan_training_tf_fns
from tensorflow_federated.python.research.gans import tff_gans
from tensorflow_federated.python.research.gans import training_loops
from tensorflow_federated.python.research.gans.experiments.emnist import emnist_data_utils
from tensorflow_federated.python.research.gans.experiments.emnist.classifier import emnist_classifier_model as ecm
from tensorflow_federated.python.research.gans.experiments.emnist.eval import emnist_eval_util as eeu
from tensorflow_federated.python.research.gans.experiments.emnist.models import convolutional_gan_networks as networks
from tensorflow_federated.python.research.gans.experiments.emnist.preprocessing import filtered_emnist_data_utils as fedu
from tensorflow_federated.python.research.utils import utils_impl

with utils_impl.record_new_flags() as hparam_flags:
  # Metadata.
  flags.DEFINE_string(
      'exp_name', 'test', 'Unique name for the experiment, suitable for use in '
      'filenames.')
  flags.DEFINE_string('root_output_dir', '/tmp/dp_fed_gan_output/',
                      'Base directory to write experimental output.')

  # Input data handling.
  flags.DEFINE_enum(
      'filtering', 'None', ['None', 'by_user', 'by_example'],
      'Indicates whether and how input EMNIST data has been filtered. If '
      '`None`, the input data has not been filtered (training will take place '
      'with all users and all examples for each user). If `by_user`, input '
      'data has been filtered down on a per-user basis; users get selected via '
      'criteria, controlled via the `invert_imagery_probability` and '
      '`accuracy_threshold` flags, and then training takes place with all '
      'examples for the selected users. If `by_example`, input data has been '
      'filtered down on a per-example basis; all users participate, and '
      'training takes place on only the examples that meet the selection '
      'criteria (controlled via the `invert_imagery_probability`, '
      '`min_num_examples`, and `example_class_selection` flags).')
  flags.DEFINE_enum(
      'invert_imagery_probability', '0p0',
      ['0p0', '0p1', '0p2', '0p3', '0p4', '0p5'],
      'The probability that a user\'s image data has pixel intensity inverted. '
      'E.g., `0p1` corresponds to 0.1, or a 10% probability that a user\'s '
      'data is flipped. Ignored if filtering is `None`.')
  flags.DEFINE_enum(
      'accuracy_threshold', 'lt0p882', ['lt0p882', 'gt0p939'],
      'When filtering is `by_user`, indicates the classification threshold by '
      'which a user is included in the training population.  E.g., `lt0p882` '
      'means any user who\'s data cumulatively classifies with <0.882 accuracy '
      'would be used for training; `gt0p939` means any user who\'s data '
      'cumulatively classifies with >0.939 accuracy would be used for '
      'training. Ignored if filtering is `None` or `by_example`.')
  flags.DEFINE_enum(
      'min_num_examples', '5', ['5', '10'],
      'When filtering is `by_example`, indicates the minimum number of '
      'examples that are correct/incorrect (as set by the flag below) in a '
      'client\'s local dataset for that client to be considered as part of '
      'training sub-population. Ignored if filtering is `None` or `by_user`.')
  flags.DEFINE_enum(
      'example_class_selection', 'incorrect', ['correct', 'incorrect'],
      'When filtering is `by_example`, indicates whether to train on a '
      'client\'s correct or incorrect examples. Ignored if filtering is `None` '
      'or `by_user`.')

  # Training hyperparameters.
  flags.DEFINE_integer(
      'num_client_disc_train_steps', 1,
      'The (max) number of optimization steps to take on each client when '
      'training the discriminator.')
  flags.DEFINE_integer(
      'num_server_gen_train_steps', 1,
      'The number of optimization steps to take on the server when training '
      'the generator.')
  flags.DEFINE_integer(
      'server_train_batch_size', 32,
      'Batch size to use on the server when training the generator.')
  flags.DEFINE_integer('num_clients_per_round', 1,
                       'The number of clients in each federated round.')
  flags.DEFINE_integer('num_rounds', 1, 'The total number of federated rounds.')
  flags.DEFINE_boolean(
      'use_dp', False,
      'If True, apply trusted aggregator user-level differential privacy. The '
      'Tensorflow Privacy library (https://github.com/tensorflow/privacy) is '
      'used, specifically the Gaussian average query. The next two flags set '
      'the specific hyperparameters of the Gaussian average query. If False, '
      'no differential privacy is applied (no clipping or noising).')
  flags.DEFINE_float(
      'dp_l2_norm_clip', 1e10,
      'If use_dp is True, the amount the discriminator weights delta vectors '
      'from the clients are clipped.')
  flags.DEFINE_float(
      'dp_noise_multiplier', 0.0,
      'If use_dp is True, Gaussian noise will be added to the sum of the '
      '(clipped) discriminator weight deltas from the clients. The standard '
      'deviation of this noise is the product of this dp_noise_multiplier and '
      'the clip parameter (dp_l2_norm_clip). Note that this noise is added '
      'before dividing by a denominator. (The denominator is the number of '
      'clients, num_clients_per_round.)')
  flags.DEFINE_float(
      'wass_gp_lambda', 10.0,
      'Value to use as the multiplier on the gradient penalty, in Improved '
      'Wasserstein GAN Loss')
  flags.DEFINE_integer('noise_dim', 128,
                       'Dimension of the generator input space.')

  # Controls output data frequency.
  flags.DEFINE_integer(
      'num_rounds_per_eval', 1,
      'The number of federated rounds to go between running evaluation.')
  flags.DEFINE_integer(
      'num_rounds_per_checkpoint', 5,
      'The number of federated rounds to go between saving a checkpoint (to be '
      'used later if a restart were necessary).')
  flags.DEFINE_integer(
      'num_rounds_per_save_images', 1,
      'The number of federated rounds to go between saving examples of '
      'generated images.')

FLAGS = flags.FLAGS

CLIENT_TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 500


def _create_gen_inputs_dataset(batch_size, noise_dim, seed=None):
  """Returns a `tf.data.Dataset` of generator random inputs."""
  return tf.data.Dataset.from_tensors(0).repeat().map(
      lambda _: tf.random.normal([batch_size, noise_dim], seed=seed))


@functools.lru_cache(maxsize=16)
def _create_real_images_dataset_for_eval():
  """Returns a `tf.data.Dataset` of real images."""
  eval_tff_data = emnist_data_utils.create_real_images_tff_client_data(
      split='test')
  raw_data = eval_tff_data.create_tf_dataset_from_all_clients()

  return emnist_data_utils.preprocess_img_dataset(
      raw_data,
      include_label=False,
      batch_size=EVAL_BATCH_SIZE,
      shuffle=True,
      repeat=True)


def _get_gan_network_models(noise_dim):
  disc_model_fn = networks.get_gan_discriminator_model

  def gen_model_fn():
    return networks.get_gan_generator_model(noise_dim)

  return disc_model_fn, gen_model_fn


def _save_images(generator, gen_inputs, outdir, file_prefix):
  """Saves images from `generator`."""
  gen_images = generator(gen_inputs, training=False)
  generated_results = np.array([
      np.reshape(gen_image.numpy(), (28, 28, 1))
      for gen_image in tf.unstack(gen_images, axis=0)[:36]
  ])

  tiled_image = tfgan.eval.python_image_grid(
      generated_results, grid_shape=(6, 6))

  if not tf.io.gfile.exists(outdir):
    tf.io.gfile.makedirs(outdir)

  f = tf.io.gfile.GFile(
      os.path.join(outdir, '{}.png'.format(file_prefix)), mode='w')
  # Convert tiled_image from float32 in [-1, 1] to uint8 [0, 255].
  pil_image = Image.fromarray(
      np.squeeze((255 / 2.0) * (tiled_image + 1.0), axis=2).astype(np.uint8))
  pil_image.convert('L').save(f, 'PNG')


def _compute_eval_metrics(generator, discriminator, gen_inputs, real_images,
                          gan_loss_fns, emnist_classifier):
  """Computes eval metrics for the GAN."""
  gen_images = generator(gen_inputs, training=False)
  disc_on_real_images = discriminator(real_images, training=False)
  disc_on_gen_outputs = discriminator(gen_images, training=False)
  real_data_logits = tf.reduce_mean(disc_on_real_images)
  gen_data_logits = tf.reduce_mean(disc_on_gen_outputs)

  gen_loss = gan_loss_fns.generator_loss(generator, discriminator, gen_inputs)
  disc_loss = gan_loss_fns.discriminator_loss(generator, discriminator,
                                              gen_inputs, real_images)
  classifier_score = eeu.emnist_score(gen_images, emnist_classifier)
  frechet_classifier_distance = eeu.emnist_frechet_distance(
      real_images, gen_images, emnist_classifier)

  metrics = collections.OrderedDict([
      ('real_data_logits', real_data_logits),
      ('gen_data_logits', gen_data_logits),
      ('gen trainable norm',
       tf.linalg.global_norm(generator.trainable_variables)),
      ('disc trainable norm',
       tf.linalg.global_norm(discriminator.trainable_variables)),
      ('gen non-trainable norm',
       tf.linalg.global_norm(generator.non_trainable_variables)),
      ('disc non-trainable norm',
       tf.linalg.global_norm(discriminator.non_trainable_variables)),
      ('gen_loss', gen_loss),
      ('disc_loss', disc_loss),
      ('classifier_score', classifier_score),
      ('frechet_classifier_distance', frechet_classifier_distance),
  ])
  return metrics


def _get_emnist_eval_hook_fn(exp_name, output_dir, hparams_dict, gan_loss_fns,
                             gen_input_eval_dataset, real_images_eval_dataset,
                             rounds_per_save_images, path_to_output_images,
                             emnist_classifier_for_metrics):
  """Returns an eval_hook function to pass to training loops."""
  tf.io.gfile.makedirs(path_to_output_images)
  logging.info(
      'Directory %s created (or already exists).', path_to_output_images)

  gen_inputs_iter = iter(gen_input_eval_dataset)
  real_images_iter = iter(real_images_eval_dataset)

  summary_logdir = os.path.join(output_dir, 'logdir/{}'.format(exp_name))
  tf.io.gfile.makedirs(summary_logdir)

  summary_writer = tf.compat.v2.summary.create_file_writer(
      summary_logdir, name=exp_name)

  # Record the hyperparameter flag settings.
  with summary_writer.as_default():
    hp.hparams(hparams_dict)

  def eval_hook(generator, discriminator, server_state, round_num):
    """Called during TFF GAN IterativeProcess, to compute eval metrics."""
    start_time = time.time()
    metrics = {}

    gen_inputs = next(gen_inputs_iter)
    real_images = next(real_images_iter)

    if round_num % rounds_per_save_images == 0:
      _save_images(
          generator,
          gen_inputs,
          outdir=path_to_output_images,
          file_prefix='emnist_tff_gen_images_step_{:05d}'.format(round_num))

    # Compute eval metrics.
    eval_metrics = _compute_eval_metrics(generator, discriminator, gen_inputs,
                                         real_images, gan_loss_fns,
                                         emnist_classifier_for_metrics)
    metrics['eval'] = eval_metrics

    # Get counters from the server_state.
    metrics['counters'] = server_state.counters

    # Write metrics to a tf.summary logdir.
    flat_metrics = tree.flatten_with_path(metrics)
    flat_metrics = [
        ('/'.join(map(str, path)), item) for path, item in flat_metrics
    ]
    flat_metrics = collections.OrderedDict(flat_metrics)
    with summary_writer.as_default():
      for name, value in flat_metrics.items():
        tf.compat.v2.summary.scalar(name, value, step=round_num)

    # Print out the counters, and log how long it took to compute/write metrics.
    for k, v in server_state.counters.items():
      print('{:>40s} {:8.0f}'.format(k, v), flush=True)
    logging.info('Doing evaluation took %.2f seconds.',
                 time.time() - start_time)

  return eval_hook


def _get_gan(gen_model_fn, disc_model_fn, gan_loss_fns, gen_optimizer,
             disc_optimizer, server_gen_inputs_dataset,
             client_real_images_tff_data, use_dp, dp_l2_norm_clip,
             dp_noise_multiplier, clients_per_round):
  """Construct instance of tff_gans.GanFnsAndTypes class."""
  dummy_gen_input = next(iter(server_gen_inputs_dataset))
  dummy_real_data = next(
      iter(
          client_real_images_tff_data.create_tf_dataset_for_client(
              client_real_images_tff_data.client_ids[0])))

  train_generator_fn = gan_training_tf_fns.create_train_generator_fn(
      gan_loss_fns, gen_optimizer)
  train_discriminator_fn = gan_training_tf_fns.create_train_discriminator_fn(
      gan_loss_fns, disc_optimizer)

  dp_average_query = None
  if use_dp:
    dp_average_query = tensorflow_privacy.GaussianAverageQuery(
        l2_norm_clip=dp_l2_norm_clip,
        sum_stddev=dp_l2_norm_clip * dp_noise_multiplier,
        denominator=clients_per_round)

  return tff_gans.GanFnsAndTypes(
      generator_model_fn=gen_model_fn,
      discriminator_model_fn=disc_model_fn,
      dummy_gen_input=dummy_gen_input,
      dummy_real_data=dummy_real_data,
      train_generator_fn=train_generator_fn,
      train_discriminator_fn=train_discriminator_fn,
      server_disc_update_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=1.0),
      train_discriminator_dp_average_query=dp_average_query)


def _train(gan, server_gen_inputs_dataset, client_gen_inputs_dataset,
           client_real_images_tff_data, client_disc_train_steps,
           server_gen_train_steps, clients_per_round, total_rounds,
           rounds_per_eval, eval_hook_fn, rounds_per_checkpoint, output_dir,
           exp_name):
  """Trains the federated GAN."""
  server_gen_inputs_iterator = iter(
      server_gen_inputs_dataset.window(server_gen_train_steps))
  client_gen_inputs_iterator = iter(
      client_gen_inputs_dataset.window(client_disc_train_steps))

  def server_gen_inputs_fn(round_num):
    del round_num
    return next(server_gen_inputs_iterator)

  def client_datasets_fn(round_num):
    """Forms clients_per_round number of datasets for a round of computation."""
    del round_num
    client_ids = np.random.choice(
        client_real_images_tff_data.client_ids,
        size=clients_per_round,
        replace=False)
    datasets = []
    for client_id in client_ids:
      datasets.append((next(client_gen_inputs_iterator),
                       client_real_images_tff_data.create_tf_dataset_for_client(
                           client_id).take(client_disc_train_steps)))
    return datasets

  return training_loops.federated_training_loop(
      gan,
      server_gen_inputs_fn=server_gen_inputs_fn,
      client_datasets_fn=client_datasets_fn,
      total_rounds=total_rounds,
      rounds_per_eval=rounds_per_eval,
      eval_hook=eval_hook_fn,
      rounds_per_checkpoint=rounds_per_checkpoint,
      root_checkpoint_dir=os.path.join(output_dir,
                                       'checkpoints/{}'.format(exp_name)))


def _get_path_to_output_image(root_output_dir, exp_name):
  path_to_output_images = os.path.join(root_output_dir,
                                       'images/{}'.format(exp_name))
  return path_to_output_images


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.set_verbosity(logging.INFO)

  # Flags.
  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])
  for k in hparam_dict.keys():
    if hparam_dict[k] is None:
      hparam_dict[k] = 'None'
  for k, v in hparam_dict.items():
    print('{} : {} '.format(k, v))

  tff.framework.set_default_executor(
      tff.framework.local_executor_factory(
          num_clients=FLAGS.num_clients_per_round))

  # Trained classifier model.
  classifier_model = ecm.get_trained_emnist_classifier_model()

  # GAN Models.
  disc_model_fn, gen_model_fn = _get_gan_network_models(FLAGS.noise_dim)

  # Training datasets.
  server_gen_inputs_dataset = _create_gen_inputs_dataset(
      batch_size=FLAGS.server_train_batch_size, noise_dim=FLAGS.noise_dim)
  client_gen_inputs_dataset = _create_gen_inputs_dataset(
      batch_size=CLIENT_TRAIN_BATCH_SIZE, noise_dim=FLAGS.noise_dim)

  if FLAGS.filtering == 'by_user':
    client_real_images_train_tff_data = (
        fedu.get_filtered_by_user_client_data_for_training(
            invert_imagery_probability=FLAGS.invert_imagery_probability,
            accuracy_threshold=FLAGS.accuracy_threshold,
            batch_size=CLIENT_TRAIN_BATCH_SIZE))
  elif FLAGS.filtering == 'by_example':
    client_real_images_train_tff_data = (
        fedu.get_filtered_by_example_client_data_for_training(
            invert_imagery_probability=FLAGS.invert_imagery_probability,
            min_num_examples=FLAGS.min_num_examples,
            example_class_selection=FLAGS.example_class_selection,
            batch_size=CLIENT_TRAIN_BATCH_SIZE))
  else:
    client_real_images_train_tff_data = (
        fedu.get_unfiltered_client_data_for_training(
            batch_size=CLIENT_TRAIN_BATCH_SIZE))

  print('There are %d unique clients that will be used for GAN training.' %
        len(client_real_images_train_tff_data.client_ids))

  # Training: GAN Losses and Optimizers.
  gan_loss_fns = gan_losses.WassersteinGanLossFns(
      grad_penalty_lambda=FLAGS.wass_gp_lambda)
  disc_optimizer = tf.keras.optimizers.SGD(lr=0.0005)
  gen_optimizer = tf.keras.optimizers.SGD(lr=0.005)

  # Eval datasets.
  gen_inputs_eval_dataset = _create_gen_inputs_dataset(
      batch_size=EVAL_BATCH_SIZE, noise_dim=FLAGS.noise_dim)
  real_images_eval_dataset = _create_real_images_dataset_for_eval()

  # Eval hook.
  path_to_output_images = _get_path_to_output_image(FLAGS.root_output_dir,
                                                    FLAGS.exp_name)
  logging.info('path_to_output_images is %s', path_to_output_images)
  eval_hook_fn = _get_emnist_eval_hook_fn(
      FLAGS.exp_name, FLAGS.root_output_dir, hparam_dict, gan_loss_fns,
      gen_inputs_eval_dataset, real_images_eval_dataset,
      FLAGS.num_rounds_per_save_images, path_to_output_images, classifier_model)

  # Form the GAN.
  gan = _get_gan(
      gen_model_fn,
      disc_model_fn,
      gan_loss_fns,
      gen_optimizer,
      disc_optimizer,
      server_gen_inputs_dataset,
      client_real_images_train_tff_data,
      use_dp=FLAGS.use_dp,
      dp_l2_norm_clip=FLAGS.dp_l2_norm_clip,
      dp_noise_multiplier=FLAGS.dp_noise_multiplier,
      clients_per_round=FLAGS.num_clients_per_round)

  # Training.
  _, tff_time = _train(
      gan,
      server_gen_inputs_dataset,
      client_gen_inputs_dataset,
      client_real_images_train_tff_data,
      FLAGS.num_client_disc_train_steps,
      FLAGS.num_server_gen_train_steps,
      FLAGS.num_clients_per_round,
      FLAGS.num_rounds,
      FLAGS.num_rounds_per_eval,
      eval_hook_fn,
      FLAGS.num_rounds_per_checkpoint,
      output_dir=FLAGS.root_output_dir,
      exp_name=FLAGS.exp_name)
  logging.info('Total training time was %4.3f seconds.', tff_time)

  print('\nTRAINING COMPLETE.')


if __name__ == '__main__':
  app.run(main)
