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
import itertools
import os
import shutil
import tempfile
from typing import Any, Mapping, Text

from absl import flags
import numpy as np
import tensorflow as tf


def iter_grid(grid_dict):
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


def atomic_write_to_csv(dataframe, output_file, overwrite=True):
  """Writes `source` to `output_file` as a (possibly zipped) CSV file.

  Args:
    dataframe: A `pandas.Dataframe` or other object with a `to_csv()` method.
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
  assert not tf.io.gfile.exists(tmp_name)
  dataframe.to_csv(tmp_name, header=True)

  # Now, copy to a temp gfile next to the final target, allowing for
  # an atomic move.
  tmp_gfile_name = os.path.join(
      os.path.dirname(output_file),
      '{}.tmp{}'.format(os.path.basename(output_file),
                        np.random.randint(0, 2**63)))
  assert not tf.io.gfile.exists(tmp_gfile_name)
  tf.io.gfile.copy(src=tmp_name, dst=tmp_gfile_name)

  # Finally, do an atomic rename and clean up:
  tf.io.gfile.rename(tmp_gfile_name, output_file, overwrite=overwrite)
  shutil.rmtree(tmp_dir)


def define_optimizer_flags(prefix: str, defaults: Mapping[Text, Any] = None):
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
    defaults: A dictionary from flag names (without prefix) to
      default values, e.g., `dict(learning_rate=0.0`)` regardless
      of prefix (see the flag names above).
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
    raise ValueError('The following defaults were not consumed:\n{}'.format(
        defaults))


def get_optimizer_from_flags(prefix: str) -> tf.keras.optimizers.Optimizer:
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
