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
"""Optimizer utilities supporting federated averaging experiments."""

import inspect
from typing import Callable

from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.research.optimization.shared import yogi


def _optimizer_canonical_name(optimizer_cls):
  """Return a short, canonical name for an optimizer for us in flags."""
  return optimizer_cls.__name__.lower()


# List of optimizers currently supported.
_SUPPORTED_OPTIMIZERS = {
    _optimizer_canonical_name(cls): cls for cls in [
        tf.keras.optimizers.SGD, tf.keras.optimizers.Adagrad,
        tf.keras.optimizers.Adam, yogi.Yogi
    ]
}


def define_optimizer_flags(prefix: str) -> None:
  """Defines flags with `prefix` to configure an optimizer.

  This method is inteded to be paired with `create_optimizer_from_flags` using
  the same `prefix`, to allow Python binaries to constructed TensorFlow
  optimizers parameterized by commandline flags.

  This creates two new flags:
    * `--<prefix>_optimizer=<optimizer name>`
    * `--<prefix>_learning_rate`

  In addition to a suite of flags for each optimizer:
    * `--<prefix>_<optimizer name>_<constructor_argument>`

  For example, given the prefix "client" this will create flags (non-exhaustive
  list):

    *  `--client_optimizer`
    *  `--client_learning_rate`
    *  `--client_sgd_momentum`
    *  `--client_sgd_nesterov`
    *  `--client_adam_beta_1`
    *  `--client_adam_beta_2`
    *  `--client_adam_epsilon`

  Then calls to `create_optimizer_from_flags('client')` will construct an
  optimizer of the type named in `--client_optimizer`, parameterized by the
  flags prefixed with the matching optimizer name. For example,  if
  `--client_optimizer=sgd`, `--client_sgd_*` flags will be used.

  IMPORTANT: For flags to be correctly parsed from the commandline, this method
  must be called before `absl.app.run(main)`, and is recommened to be called
  next to other flag definitions at the top of a py_binary.

  Args:
    prefix: A string (possibly empty) indicating which optimizer is being
      configured.
  """
  # Create top-level, non-optimizer specific flags for picking the optimizer
  # type and the learning rate.
  flags.DEFINE_enum(
      name='{!s}_optimizer'.format(prefix),
      default=None,
      enum_values=list(_SUPPORTED_OPTIMIZERS.keys()),
      help='The type of optimizer to construct for `{!s}`'.format(prefix))
  logging.info('Defined new flag: [%s]', '{!s}_optimizer'.format(prefix))
  flags.DEFINE_float(
      name='{!s}_learning_rate'.format(prefix),
      default=None,
      help='Learning rate for optimizer `{!s}`'.format(prefix))
  logging.info('Defined new flag: [%s]', '{!s}_learning_rate'.format(prefix))

  for optimizer_name, optimizer_cls in _SUPPORTED_OPTIMIZERS.items():
    # Pull out the constructor parameters except for `self`.
    constructor_signature = inspect.signature(optimizer_cls.__init__)
    constructor_params = list(constructor_signature.parameters.values())[1:]

    def prefixed(basename, optimizer_name=optimizer_name):
      if prefix:
        return '{!s}_{!s}_{!s}'.format(prefix, optimizer_name, basename)
      else:
        return '{!s}_{!s}'.format(optimizer_name, basename)

    for param in constructor_params:
      if param.name in ['kwargs', 'args', 'learning_rate']:
        continue

      if isinstance(param.default, bool):
        define_flag_fn = flags.DEFINE_bool
      elif isinstance(param.default, float):
        define_flag_fn = flags.DEFINE_float
      elif isinstance(param.default, int):
        define_flag_fn = flags.DEFINE_integer
      elif isinstance(param.default, str):
        define_flag_fn = flags.DEFINE_string
      else:
        raise NotImplementedError('Cannot handle flag [{!s}] of type [{!s}] on '
                                  'optimizers [{!s}]'.format(
                                      param.name, type(param.default),
                                      optimizer_name))
      define_flag_fn(
          name=prefixed(param.name),
          default=param.default,
          help='{!s} argument for the {!s} optimizer.'.format(
              param.name, optimizer_name))
      logging.info('Defined new flag: [%s]', prefixed(param.name))


def create_optimizer_fn_from_flags(
    prefix: str) -> Callable[[float], tf.keras.optimizers.Optimizer]:
  """Returns an optimizer function based on prefixed flags.

  This method is inteded to be paired with `define_optimizer_flags` using the
  same `prefix`, to allow Python binaries to constructed TensorFlow optimizers
  parameterized by commandline flags.

  This method expects at least two flags to have been defined and set:
    * `--<prefix>_optimizer=<optimizer name>`
    * `--<prefix>_learning_rate`

  In addition to suites of flags for each optimizer:
    * `--<prefix>_<optimizer name>_<constructor_argument>`

  For example, if `prefix='client'` this method first reads the flags:
    * `--client_optimizer`
    * `--client_learning_rate`

  If the optimizer flag is `'sgd'`, then a `tf.keras.optimizer.SGD` optimizer is
  constructed using the values in the flags prefixed with  `--client_sgd_`.

  Args:
    prefix: The same string prefix passed to `define_optimizer_flags`.

  Returns:
    A 1-arg function that accepts a learning rate and returns a
      `tf.keras.optimizers.Optimizer`.
  """
  def prefixed(basename):
    return '{}_{}'.format(prefix, basename) if prefix else basename

  optimizer_flag_name = prefixed('optimizer')
  if flags.FLAGS[optimizer_flag_name] is None:
    raise ValueError('Must specify flag --{!s}'.format(optimizer_flag_name))
  optimizer_name = flags.FLAGS[optimizer_flag_name].value
  optimizer_cls = _SUPPORTED_OPTIMIZERS.get(optimizer_name)
  if optimizer_cls is None:
    # To support additional optimizers, implement it as a
    # `tf.keras.optimizers.Optimizer` and add to the `_SUPPORTED_OPTIMIZERS`
    # dict.
    logging.error(
        'Unknown optimizer [%s], known optimziers are [%s]. To add '
        'support for an optimizer, add the optimzier class to the '
        'utils_impl._SUPPORTED_OPTIMIZERS list.', optimizer_name,
        list(_SUPPORTED_OPTIMIZERS.keys()))
    raise ValueError('`{!s}` is not a valid optimizer for flag --{!s}, must be '
                     'one of {!s}. See error log for details.'.format(
                         optimizer_name, optimizer_flag_name,
                         list(_SUPPORTED_OPTIMIZERS.keys())))

  def _has_user_value(flag):
    """Check if a commandline flag has a user set value."""
    return flag.present or flag.value != flag.default

  # Validate that the optimizers that weren't picked don't have flag values set.
  # Settings that won't be used likely means there is an expectation gap between
  # the user and the system and we should notify them.
  unused_flag_prefixes = [
      prefixed(k) for k in _SUPPORTED_OPTIMIZERS.keys() if k != optimizer_name
  ]
  mistakenly_set_flags = []
  for flag_name in flags.FLAGS:
    if not _has_user_value(flags.FLAGS[flag_name]):
      # Flag was not set by the user, skip it.
      continue
    # Otherwise the flag has a value set by the user.
    for unused_prefix in unused_flag_prefixes:
      if flag_name.startswith(unused_prefix):
        mistakenly_set_flags.append(flag_name)
        break
  if mistakenly_set_flags:
    raise ValueError('Commandline flags for optimizers other than [{!s}] '
                     '(value of --{!s}) are set. These would be ignored, '
                     'were the flags set by mistake? Flags: {!s}'.format(
                         optimizer_name, optimizer_flag_name,
                         mistakenly_set_flags))

  lr_flag_name = prefixed('learning_rate')
  lr_flag = flags.FLAGS[lr_flag_name]
  if _has_user_value(lr_flag):
    default_lr = lr_flag.value
  else:
    raise ValueError(
        'Learning rate for {!s} must be set by the flag --{!s} .'.format(
            prefix, lr_flag_name))

  flag_prefix = prefixed(optimizer_name)
  prefix_len = len(flag_prefix) + 1
  kwargs = {}
  for flag_name in flags.FLAGS:
    if not flag_name.startswith(flag_prefix):
      continue
    arg_name = flag_name[prefix_len:]
    kwargs[arg_name] = flags.FLAGS[flag_name].value

  if 'learning_rate' in kwargs:
    kwargs.pop('learning_rate')

  def optimizer_fn(learning_rate=default_lr):
    return optimizer_cls(learning_rate=learning_rate, **kwargs)
  return optimizer_fn


def define_lr_schedule_flags(prefix: str) -> None:
  """Defines flags with `prefix` to configure a learning rate schedule.

  This method is intended to be paired with `create_optimizer_from_flags` with
  the same `prefix`, to allow Python binaries to construct `tf.keras.optimizer`
  objects from flags, along with an associated learning rate schedule.

  This creates four new flags:
    * `--<prefix>_lr_schedule`
    * `--<prefix>_lr_decay_step`
    * `--<prefix>_lr_decay_rate`
    * `--<prefix>_lr_staircase`

  Note that this should generally be preceded by `define_optimizer_flags`, and
  followed by `create_lr_schedule_from_flags`. This will then create a learning
  rate scheduling function governed by the flags defined herein.

  Args:
    prefix: A string (possibly empty) indicating which optimizer is being
      configured.
  """
  def prefixed(basename):
    return '{}_{}'.format(prefix, basename) if prefix else basename

  initial_lr_flag_name = prefixed('learning_rate')
  if flags.FLAGS[initial_lr_flag_name] is None:
    logging.warning(
        'The flag %s is not set. This must be set before calling '
        '`create_lr_schedule_from_flags`.', initial_lr_flag_name)

  flags.DEFINE_enum(
      '{!s}_lr_schedule'.format(prefix),
      default='constant',
      enum_values=['constant', 'exp_decay', 'inv_lin_decay', 'inv_sqrt_decay'],
      help='Type of learning rate decay schedule to use for `{!s}`.'.format(
          prefix))
  flags.DEFINE_integer(
      '{!s}_lr_decay_steps'.format(prefix),
      default=None,
      help='An int used to compute the learning rate schedule.'
      'If staircase is set to True, then the learning rate changes every '
      '`{!s}_lr_decay_steps` rounds.'.format(prefix))
  flags.DEFINE_float(
      '{!s}_lr_decay_rate'.format(prefix),
      default=None,
      help='The decay rate of the {!s} learning rate schedule.'.format(prefix))
  flags.DEFINE_bool(
      '{!s}_lr_staircase'.format(prefix),
      default=False,
      help='Whether to decay the `{!s}` learning rate at discrete intervals.'
      .format(prefix))


def exp_decay_schedule_builder(base_value, decay_steps, decay_rate, staircase):
  """Creates a learning rate schedule with exponential root decay.

  Args:
    base_value: The initial value of the quantity to decay over time.
    decay_steps: A positive scalar that governs how much the value decays at a
      given round number.
    decay_rate: A float between 0 and 1 that governs how quickly the decay
      occurs.
    staircase: A boolean. If set to True, the decaying occurs in discrete
      intervals.

  Returns:
    A 1-arg callable that produces a decayed version of the base value when
      passed the current round_num.
  """
  if staircase:
    def exp_decay_fn(round_num):
      return base_value * tf.pow(decay_rate, round_num // decay_steps)
  else:
    def exp_decay_fn(round_num):
      return base_value * tf.pow(decay_rate, round_num / decay_steps)

  return exp_decay_fn


def inv_lin_schedule_builder(base_value, decay_steps, decay_rate, staircase):
  """Creates a learning rate schedule with inverse linear decay.

  Args:
    base_value: The initial value of the quantity to decay over time.
    decay_steps: A positive scalar that governs how much the value decays at a
      given round number.
    decay_rate: A positive scalar that governs how quickly the decay occurs.
    staircase: A boolean. If set to True, the decaying occurs in discrete
      intervals.

  Returns:
    A 1-arg callable that produces a decayed version of the base value when
      passed the current round_num.
  """
  if staircase:
    def inv_lin_decay_fn(round_num):
      return base_value / (1.0 + decay_rate * (round_num // decay_steps))
  else:
    def inv_lin_decay_fn(round_num):
      return base_value / (1.0 + decay_rate * (round_num / decay_steps))

  return inv_lin_decay_fn


def inv_sqrt_schedule_builder(base_value, decay_steps, decay_rate, staircase):
  """Creates a learning rate schedule with inverse square root decay.

  Args:
    base_value: The initial value of the quantity to decay over time.
    decay_steps: A positive scalar that governs how much the value decays at a
      given round number.
    decay_rate: A positive scalar that governs how quickly the decay occurs.
    staircase: A boolean. If set to True, the decaying occurs in discrete
      intervals.

  Returns:
    A 1-arg callable that produces a decayed version of the base value when
      passed the current round_num.
  """
  if staircase:
    def inv_sqrt_decay_fn(round_num):
      return base_value / tf.sqrt(1.0 + decay_rate * (round_num // decay_steps))
  else:
    def inv_sqrt_decay_fn(round_num):
      return base_value / tf.sqrt(1.0 + decay_rate * (round_num / decay_steps))

  return inv_sqrt_decay_fn


def create_lr_schedule_from_flags(
    prefix: str) -> Callable[[tf.Tensor], tf.Tensor]:
  """Returns a callable learning rate schedule based on prefix flags.

  This method is inteded to be paired with `define_lr_schedule_flags` using the
  same `prefix`, to construct a callable learning rate schedule parameterized by
  commandline flags.

  This method expects the following flags to have been defined and set:
    * `--<prefix>_learning_rate`
    * `--<prefix>_lr_schedule`

  If `<prefix>_lr_schedule` is set to `constant`, then this function will
  return a callable that always outputs <prefix>_learning_rate`.

  If <prefix>_lr_schedule is not `constant`, then this method expects the
    following flags to be defined as well:
    * `--<prefix>_lr_decay_steps`
    * `--<prefix>_lr_decay_rate`
    * `--<prefix>_lr_staircase

  Args:
    prefix: The same string prefix passed to `define_optimizer_flags`.

  Returns:
    A callable that accepts a `round_num` and returns a learning rate.
  """

  def prefixed(basename):
    return '{}_{}'.format(prefix, basename) if prefix else basename

  lr_flag_name = prefixed('learning_rate')
  if flags.FLAGS[lr_flag_name] is None:
    raise ValueError('Must specify flag --{!s}'.format(lr_flag_name))
  lr_schedule_flag_name = prefixed('lr_schedule')
  if flags.FLAGS[lr_schedule_flag_name] is None:
    raise ValueError('Must specify flag --{!s}'.format(lr_schedule_flag_name))

  base_lr = flags.FLAGS[lr_flag_name].value
  lr_schedule_type = flags.FLAGS[lr_schedule_flag_name].value
  if lr_schedule_type == 'constant':
    return lambda round_num: base_lr

  lr_decay_steps = flags.FLAGS[prefixed('lr_decay_steps')].value
  lr_decay_rate = flags.FLAGS[prefixed('lr_decay_rate')].value
  lr_staircase = flags.FLAGS[prefixed('lr_staircase')].value

  if lr_schedule_type == 'exp_decay':
    return exp_decay_schedule_builder(
        base_lr, lr_decay_steps, lr_decay_rate, lr_staircase)
  elif lr_schedule_type == 'inv_lin_decay':
    return inv_lin_schedule_builder(
        base_lr, lr_decay_steps, lr_decay_rate, lr_staircase)
  elif lr_schedule_type == 'inv_sqrt_decay':
    return inv_sqrt_schedule_builder(
        base_lr, lr_decay_steps, lr_decay_rate, lr_staircase)
  else:
    raise ValueError(
        'Unrecognized schedule type {!s}'.format(lr_schedule_type))


def compute_yogi_init(dataset, model, num_clients):
  """Computes an estimate for the Yogi initial accumulator.

  Args:
    dataset: A `tff.simulation.ClientData` object.
    model: A `tff.learning.Model` object.
    num_clients: The number of clients to sample batches from.

  Returns:
    A float estimate of the per-parameter Yogi initial accumulator value.
  """
  num_trainable_params = np.sum(
      [np.prod(v.get_shape()) for v in model.trainable_variables])

  client_list = dataset.client_ids
  np.random.shuffle(client_list)
  grad_norm_estimate = 0.0
  num_clients_sampled = 0

  @tf.function
  def compute_batch_gradient_l2_norm_squared(model, batch):
    with tf.GradientTape() as tape:
      output = model.forward_pass(batch)
    grads = tape.gradient(output.loss, model.trainable_variables)
    grad_norms = tf.nest.map_structure(lambda x: tf.linalg.norm(x)**2, grads)
    total_grad_norm = tf.reduce_sum(grad_norms)
    return total_grad_norm

  for client_id in client_list:
    try:
      batch = next(iter(dataset.create_tf_dataset_for_client(client_id)))
      grad_norm_estimate += compute_batch_gradient_l2_norm_squared(
          model, batch).numpy() / num_trainable_params
      num_clients_sampled += 1
      if num_clients_sampled == num_clients:
        break
    except StopIteration:
      pass  # Client had no batches.
  return grad_norm_estimate / num_clients_sampled
