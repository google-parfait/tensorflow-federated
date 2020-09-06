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
"""Library for shared Keras callbacks."""
import os.path
from typing import Dict, Any

import pandas as pd
import tensorflow as tf

from tensorflow_federated_research.utils import utils_impl


class AtomicCSVLogger(tf.keras.callbacks.Callback):
  """A callback that writes per-epoch values to a CSV file."""

  def __init__(self, path: str):
    self._path = path

  def on_epoch_end(self, epoch: int, logs: Dict[Any, Any] = None):
    results_path = os.path.join(self._path, 'metric_results.csv')
    if tf.io.gfile.exists(results_path):
      # Read the results until now.
      results_df = utils_impl.atomic_read_from_csv(results_path)
      # Slice off results after the current epoch, this indicates the job
      # restarted.
      results_df = results_df[:epoch]
      # Add the new epoch.
      results_df = results_df.append(logs, ignore_index=True)
    else:
      results_df = pd.DataFrame(logs, index=[epoch])
    utils_impl.atomic_write_to_csv(results_df, results_path)
