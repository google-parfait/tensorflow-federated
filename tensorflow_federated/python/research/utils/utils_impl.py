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
"""Utilities supporting experiments."""

import collections
import contextlib
import functools
import itertools
import multiprocessing
import os
import shutil
import subprocess
import tempfile

from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Text, Union
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf


def iter_grid(
    grid_dict: Mapping[Text, Sequence[Union[int, float, Text]]]
) -> Iterator[Dict[Text, Union[int, float, Text]]]:
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
                        output_file: Text,
                        overwrite: bool = True) -> None:
  """Writes `source` to `output_file` as a (possibly zipped) CSV file.

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


def define_optimizer_flags(
    prefix: Text,
    defaults: Optional[Dict[Text, Union[int, float, Text]]] = None) -> None:
  """Defines flags with `prefix` to configure an optimizer.

  For flags to be correctly parsed, this should be called next to other
  flag definitions at the top of a py_binary, before `absl.app.run(main)`.

  For example, given the prefix "client" this will create flags:

    *  `--client_learning_rate`

  It is expected that more flags to support other optimizers will be added
  in the future. Currently, the optimizer is always SGD.

  Args:
    prefix: A string (possibly empty) indicating which optimizer is being
      configured.
    defaults: A dictionary from flag names (without prefix) to default values,
      e.g., `dict(learning_rate=0.0`)` regardless of prefix (see the flag names
      above).
  """

  # Note: An alternative design is to use a single flag for each optimizer,
  # something like:
  # --server_optimizer="name=SGD,learning_rate=1.0" or
  # --client_optimizer="name=ADAM,learning_rate_=0.001,beta1=0.9"
  #
  # These options could potentially follow the structure of the
  # optimizer.get_config() method. This has the advantage over the current
  # approach that there won't be flags that could be silently ignored (e.g.,
  # beta1 when the optimizer is SGD). However, it may not work as well for
  # systems that want one flag equals one hyperparameter. It would also require
  # writing custom parsing and validation code.

  defaults = defaults or {}

  def prefixed(basename):
    return '{}_{}'.format(prefix, basename) if prefix else basename

  opt_name = 'the {} optimizer'.format(prefix) if prefix else 'the optimizer'

  # More flags can be added here, e.g. for momentum, Adam, etc.
  # For now we just assume SGD.
  flags.DEFINE_float(
      prefixed('learning_rate'), defaults.pop('learning_rate', 0.1),
      'Learning rate for {}'.format(opt_name))

  if defaults:  # Not empty.
    raise ValueError(
        'The following defaults were not consumed:\n{}'.format(defaults))


def get_optimizer_from_flags(prefix: Text) -> tf.keras.optimizers.Optimizer:
  """Returns an optimizer based on flags defined by `define_optimzier_flags`.

  Args:
    prefix: The same string prefix passed to `define_optimizer_flags`.

  Returns:
    A `tf.keras.optimizers.Optimizer`.
  """

  def flag_value(basename):
    full_name = '{}_{}'.format(prefix, basename) if prefix else basename
    return flags.FLAGS[full_name].value

  return tf.keras.optimizers.SGD(learning_rate=flag_value('learning_rate'))


@contextlib.contextmanager
def record_new_flags() -> Iterator[List[Text]]:
  """A context manager that returns all flags created in it's scope.

  This is useful to define all of the flags which should be considered
  hyperparameters of the training run, without needing to repeat them.

  Example usage:
  ```python
  with record_new_flags() as hparam_flags:
      flags.DEFINE_string('exp_name', 'name', 'Unique name for the experiment.')
      flags.DEFINE_integer('random_seed', 0, 'Random seed for the experiment.')
  ```

  Check `research/emnist/run_experiment.py` for more details about the usage.

  Yields:
    A list of all newly created flags.
  """
  old_flags = set(iter(flags.FLAGS))
  new_flags = []
  yield new_flags
  new_flags.extend([f for f in flags.FLAGS if f not in old_flags])


def hparams_to_str(wid: int,
                   param_dict: Mapping[Text, Text],
                   short_names: Optional[Mapping[Text, Text]] = None) -> Text:
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


def launch_experiment(executable: Text,
                      grid_iter: Iterable[Mapping[Text, Union[int, float,
                                                              Text]]],
                      root_output_dir: Text = '/tmp/exp',
                      short_names: Optional[Mapping[Text, Text]] = None,
                      max_workers: int = 1):
  """Launch experiments of grid search in parallel or sequentially.

  Example usage:
  ```python
  grid_iter = iter_grid({'a': [1, 2], 'b': [4.0, 5.0]))
  launch_experiment('run_exp.py', grid_iter)
  ```

  Args:
    executable: An executable python file which takes flags --root_output_dir
      and --exp_name, e.g., `research/emnist/run_experiment.py`.
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
    command = 'python {} {}'.format(executable, ' '.join(param_list))
    command_list.append(command)

  pool = multiprocessing.Pool(processes=max_workers)
  executor = functools.partial(subprocess.call, shell=True)
  for command in command_list:
    pool.apply_async(executor, (command,))
  pool.close()
  pool.join()
