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
"""Utilities supporting experiments."""

import collections
import contextlib
import functools
import inspect
import itertools
import multiprocessing
import os.path
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf


def iter_grid(
    grid_dict: Mapping[str, Sequence[Union[int, float, str]]]
) -> Iterator[Dict[str, Union[int, float, str]]]:
  """Iterates over all combinations of values in the provied dict-of-lists.

  >>> list(iter_grid({'a': [1, 2], 'b': [4.0, 5.0, 6.0]))
  [OrderedDict([('a', 1), ('b', 4.0)]),
   OrderedDict([('a', 1), ('b', 5.0)]),
   OrderedDict([('a', 1), ('b', 6.0)]),
   OrderedDict([('a', 2), ('b', 4.0)]),
   OrderedDict([('a', 2), ('b', 5.0)]),
   OrderedDict([('a', 2), ('b', 6.0)])]

  Args:
    grid_dict: A dictionary of iterables.

  Yields:
    A sequence of dictionaries with keys from grid, and values corresponding
    to all combinations of items in the corresponding iterables.
  """
  names_to_lists = collections.OrderedDict(sorted(grid_dict.items()))
  names = names_to_lists.keys()
  for values in itertools.product(*names_to_lists.values()):
    yield collections.OrderedDict(zip(names, values))


def atomic_write_to_csv(dataframe: pd.DataFrame,
                        output_file: str,
                        overwrite: bool = True) -> None:
  """Writes `dataframe` to `output_file` as a (possibly zipped) CSV file.

  Args:
    dataframe: A `pandas.Dataframe`.
    output_file: The final output file to write. The output will be compressed
      depending on the filename, see documentation for
      pandas.DateFrame.to_csv(compression='infer').
    overwrite: Whether to overwrite output_file if it exists.
  """
  # Exporting via to_hdf() is an appealing option, because we could perhaps
  # maintain more type information, and also write both hyperparameters and
  # results to the same HDF5 file. However, to_hdf() call uses pickle under the
  # hood, and there seems to be no way to tell it to use pickle protocol=2, it
  # defaults to 4. This means the results cannot be read from Python 2. We
  # currently still want Python 2 support, so sticking with CSVs for now.

  # At least when writing a zip, .to_csv() is not happy taking a gfile,
  # so we need a temp file on the local filesystem.
  tmp_dir = tempfile.mkdtemp(prefix='atomic_write_to_csv_tmp')
  # We put the output_file name last so we preserve the extension to allow
  # inference of the desired compression format. Note that files with .zip
  # extension (but not .bz2, .gzip, or .xv) have unexpected internal filenames
  # due to https://github.com/pandas-dev/pandas/issues/26023, not
  # because of something we are doing here.
  tmp_name = os.path.join(tmp_dir, os.path.basename(output_file))
  assert not tf.io.gfile.exists(tmp_name), 'file [{!s}] exists'.format(tmp_name)
  dataframe.to_csv(tmp_name, header=True)

  # Now, copy to a temp gfile next to the final target, allowing for
  # an atomic move.
  tmp_gfile_name = os.path.join(
      os.path.dirname(output_file), '{}.tmp{}'.format(
          os.path.basename(output_file), np.random.randint(0, 2**63)))
  tf.io.gfile.copy(src=tmp_name, dst=tmp_gfile_name, overwrite=overwrite)

  # Finally, do an atomic rename and clean up:
  tf.io.gfile.rename(tmp_gfile_name, output_file, overwrite=overwrite)
  shutil.rmtree(tmp_dir)


def atomic_read_from_csv(csv_file):
  """Reads a `pandas.DataFrame` from the (possibly zipped) `csv_file`.

  Format note: The CSV is expected to have an index column.

  Args:
    csv_file: A (possibly zipped) CSV file.

  Returns:
    A `pandas.Dataframe`.
  """

  # When reading from a zip, pandas.from_csv() is not happy taking a gfile,
  # so we need a temp file on the local filesystem.
  tmp_dir = tempfile.mkdtemp(prefix='atomic_read_from_csv_tmp')
  # We put the output_file name last so we preserve the extension to allow
  # inference of the compression format. Note that files with .zip extension
  # (but not .bz2, .gzip, or .xv) have unexpected internal filenames due to
  # https://github.com/pandas-dev/pandas/issues/26023, not because of something
  # we are doing here.
  tmp_name = os.path.join(tmp_dir, os.path.basename(csv_file))
  assert not tf.io.gfile.exists(tmp_name), 'file [{!s}] exists'.format(tmp_name)
  tf.io.gfile.copy(src=csv_file, dst=tmp_name, overwrite=True)
  # Do the read from the temp file.
  dataframe = pd.read_csv(tmp_name, index_col=0)
  # Finally, clean up:
  shutil.rmtree(tmp_dir)

  return dataframe


def _optimizer_canonical_name(optimizer_cls):
  """Return a short, canonical name for an optimizer for us in flags."""
  return optimizer_cls.__name__.lower()


# List of optimizers currently supported.
_SUPPORTED_OPTIMIZERS = {
    _optimizer_canonical_name(cls): cls for cls in [
        tf.keras.optimizers.SGD, tf.keras.optimizers.Adagrad,
        tf.keras.optimizers.Adam
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

  Note: This method does not create a flag for `kwargs` of the Optimizer
  constructor. However, `kwargs` can be set using the `overrides` parameter of
  `create_optimizer_from_flags` below.

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


def create_optimizer_from_flags(
    prefix: str,
    overrides: Optional[Mapping[str, Union[str, float, int, bool]]] = None
) -> tf.keras.optimizers.Optimizer:
  """Returns an optimizer based on prefixed flags.

  This method is inteded to be paired with `define_optimizer_flags` using the
  same `prefix`, to allow Python binaries to constructed TensorFlow optimizers
  parameterized by commandline flags.

  This method expects at least two flags to have been defined:
    * `--<prefix>_optimizer=<optimizer name>`
    * `--<prefix>_learning_rate`

  In addition to suites of flags for each optimizer:
    * `--<prefix>_<optimizer name>_<constructor_argument>`

  For example, if `prefix='client'` this method first reads the flags:
    * `--client_optimizer`
    * `--client_learning_rate`

  If the optimizer flag is `'sgd'`, then a `tf.keras.optimizer.SGD` optimizer is
  constructed using the values in the flags prefixed with  `--client_sgd_`.

  Note: `kwargs` can be set using the `overrides` parameter.

  Args:
    prefix: The same string prefix passed to `define_optimizer_flags`.
    overrides: A mapping of `(string, value)` pairs that should override default
      flag values (but not user specified values from the commandline).

  Returns:
    A `tf.keras.optimizers.Optimizer`.
  """
  if overrides is not None:
    if not isinstance(overrides, collections.Mapping):
      raise TypeError(
          '`overrides` must be a value of type `collections.Mapping`, '
          'found type: {!s}'.format(type(overrides)))
  else:
    overrides = {}

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

  flag_prefix = prefixed(optimizer_name)
  prefix_len = len(flag_prefix) + 1
  kwargs = dict(overrides) if overrides is not None else {}
  learning_rate_flag = flags.FLAGS[prefixed('learning_rate')]
  if _has_user_value(learning_rate_flag):
    kwargs['learning_rate'] = learning_rate_flag.value
  for flag_name in flags.FLAGS:
    if not flag_name.startswith(flag_prefix):
      continue
    arg_name = flag_name[prefix_len:]
    kwargs[arg_name] = flags.FLAGS[flag_name].value
  return optimizer_cls(**kwargs)


def remove_unused_flags(prefix, hparam_dict):
  """Removes unused optimizer flags with a given prefix.

  This method is intended to be used with `define_optimizer_flags`, and is used
  to remove elements of hparam_dict associated with unused optimizer flags.

  For example, given the prefix "client", define_optimizer_flags will create
  flags including:
    *  `--client_optimizer`
    *  `--client_learning_rate`
    *  `--client_sgd_momentum`
    *  `--client_sgd_nesterov`
    *  `--client_adam_beta_1`
    *  `--client_adam_beta_2`
    *  `--client_adam_epsilon`

  However, for purposes of recording hyperparameters, we would like to only keep
  those that correspond to the optimizer selected in the flag
  --client_optimizer. This method is intended to remove the unused flags.

  For example, if `--client_optimizer=sgd` was set, then calling this method
  with the prefix `client` will remove all pairs in hparam_dict except those
  associated with the flags:
    *  `--client_optimizer`
    *  `--client_learning_rate`
    *  `--client_sgd_momentum`
    *  `--client_sgd_nesterov`

  Args:
    prefix: A prefix used to define optimizer flags.
    hparam_dict: An ordered dictionary of (string, value) pairs corresponding to
      experiment hyperparameters.

  Returns:
    An ordered dictionary of (string, value) pairs from hparam_dict that omits
    any pairs where string = "<prefix>_<optimizer>*" but <optimizer> is not the
    one set via the flag --<prefix>_optimizer=...
  """

  def prefixed(basename):
    return '{}_{}'.format(prefix, basename) if prefix else basename

  if prefixed('optimizer') not in hparam_dict.keys():
    raise ValueError('The flag {!s} was not defined.'.format(
        prefixed('optimizer')))

  optimizer_name = hparam_dict[prefixed('optimizer')]
  if not optimizer_name:
    raise ValueError('The flag {!s} was not set. Unable to determine the '
                     'relevant optimizer.'.format(prefixed('optimizer')))

  unused_optimizer_flag_prefixes = [
      prefixed(k) for k in _SUPPORTED_OPTIMIZERS.keys() if k != optimizer_name
  ]

  def _is_used_flag(flag_name):
    # We filter by whether the flag contains an unused optimizer prefix.
    # This automatically retains any flag not of the form <prefix>_<optimizer>*.
    for unused_flag_prefix in unused_optimizer_flag_prefixes:
      if flag_name.startswith(unused_flag_prefix):
        return False
    return True

  return collections.OrderedDict([
      (flag_name, flag_value)
      for flag_name, flag_value in hparam_dict.items()
      if _is_used_flag(flag_name)
  ])


_all_hparam_flags = []


@contextlib.contextmanager
def record_hparam_flags():
  """A context manager that adds all flags created in its scope to a global list of flags, and yields all flags created in its scope.

  This is useful for defining hyperparameter flags of an experiment, especially
  when the flags are partitioned across a number of modules. The total list of
  flags defined across modules can then be accessed via get_hparam_flags().

  Example usage:
  ```python
  with record_hparam_flags() as optimizer_hparam_flags:
      flags.DEFINE_string('optimizer', 'sgd', 'Optimizer for training.')
  with record_hparam_flags() as evaluation_hparam_flags:
      flags.DEFINE_string('eval_metric', 'accuracy', 'Metric for evaluation.')
  experiment_hparam_flags = get_hparam_flags().
  ```

  Check `research/optimization/emnist/run_emnist.py` for more usage details.

  Yields:
    A list of all newly created flags.
  """
  old_flags = set(iter(flags.FLAGS))
  new_flags = []
  yield new_flags
  new_flags.extend([f for f in flags.FLAGS if f not in old_flags])
  _all_hparam_flags.extend(new_flags)


def get_hparam_flags():
  """Returns a list of flags defined within the scope of record_hparam_flags."""
  return _all_hparam_flags


@contextlib.contextmanager
def record_new_flags() -> Iterator[List[str]]:
  """A context manager that returns all flags created in its scope.

  This is useful to define all of the flags which should be considered
  hyperparameters of the training run, without needing to repeat them.

  Example usage:
  ```python
  with record_new_flags() as hparam_flags:
      flags.DEFINE_string('exp_name', 'name', 'Unique name for the experiment.')
  ```

  Check `research/emnist/run_experiment.py` for more details about the usage.

  Yields:
    A list of all newly created flags.
  """
  old_flags = set(iter(flags.FLAGS))
  new_flags = []
  yield new_flags
  new_flags.extend([f for f in flags.FLAGS if f not in old_flags])


def lookup_flag_values(flag_list: Iterable[str]) -> collections.OrderedDict:
  """Returns a dictionary of (flag_name, flag_value) pairs for an iterable of flag names."""
  flag_odict = collections.OrderedDict()
  for flag_name in flag_list:
    if not isinstance(flag_name, str):
      raise ValueError(
          'All flag names must be strings. Flag {} was of type {}.'.format(
              flag_name, type(flag_name)))

    if flag_name not in flags.FLAGS:
      raise ValueError('"{}" is not a defined flag.'.format(flag_name))
    flag_odict[flag_name] = flags.FLAGS[flag_name].value

  return flag_odict


def hparams_to_str(wid: int,
                   param_dict: Mapping[str, str],
                   short_names: Optional[Mapping[str, str]] = None) -> str:
  """Convenience method which flattens the hparams to a string.

  Used as mapping function for the WorkUnitCustomiser.

  Args:
    wid: Work unit id, int type.
    param_dict: A dict of parameters.
    short_names: A dict of mappings of parameter names.

  Returns:
    The hparam string.
  """
  if not param_dict:
    return str(wid)

  if not short_names:
    short_names = {}

  name = [
      '{}={}'.format(short_names.get(k, k), str(v))
      for k, v in sorted(param_dict.items())
  ]
  hparams_str = '{}-{}'.format(str(wid), ','.join(name))

  # Escape some special characters
  replace_str = {
      '\n': ',',
      ':': '=',
      '\'': '',
      '"': '',
  }
  for c, new_c in replace_str.items():
    hparams_str = hparams_str.replace(c, new_c)
  for c in ('\\', '/', '[', ']', '(', ')', '{', '}', '%'):
    hparams_str = hparams_str.replace(c, '-')
  if len(hparams_str) > 170:
    raise ValueError(
        'hparams_str string is too long ({}). You can input a short_name dict '
        'to map the long parameter name to a short name. For example, '
        ' launch_experiment(executable, grid_iter, '
        ' {{server_learning_rate: s_lr}}) \n'
        'Received: {}'.format(len(hparams_str), hparams_str))
  return hparams_str


def launch_experiment(executable: str,
                      grid_iter: Iterable[Mapping[str, Union[int, float, str]]],
                      root_output_dir: str = '/tmp/exp',
                      short_names: Optional[Mapping[str, str]] = None,
                      max_workers: int = 1):
  """Launch experiments of grid search in parallel or sequentially.

  Example usage:
  ```python
  grid_iter = iter_grid({'a': [1, 2], 'b': [4.0, 5.0]))
  launch_experiment('run_exp.py', grid_iter)
  ```

  Args:
    executable: An executable which takes flags --root_output_dir
      and --exp_name, e.g., `bazel run //research/emnist:run_experiment --`.
    grid_iter: A sequence of dictionaries with keys from grid, and values
      corresponding to all combinations of items in the corresponding iterables.
    root_output_dir: The directory where all outputs are stored.
    short_names: Short name mapping for the parameter name used if parameter
      string length is too long.
    max_workers: The max number of commands to run in parallel.
  """
  command_list = []
  for idx, param_dict in enumerate(grid_iter):
    param_list = [
        '--{}={}'.format(key, str(value))
        for key, value in sorted(param_dict.items())
    ]

    short_names = short_names or {}
    param_str = hparams_to_str(idx, param_dict, short_names)

    param_list.append('--root_output_dir={}'.format(root_output_dir))
    param_list.append('--exp_name={}'.format(param_str))
    command = '{} {}'.format(executable, ' '.join(param_list))
    command_list.append(command)

  pool = multiprocessing.Pool(processes=max_workers)
  executor = functools.partial(subprocess.call, shell=True)
  for command in command_list:
    pool.apply_async(executor, (command,))
  pool.close()
  pool.join()
