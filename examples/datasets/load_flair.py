# Copyright 2023, The TensorFlow Federated Authors.
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
"""An example script that loads and processes the FLAIR dataset."""

import collections
from collections.abc import Sequence
import tempfile

from absl import app
from absl import logging
import tensorflow_federated as tff


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  data_dir = tempfile.mkdtemp()
  tff.simulation.datasets.flair.download_data(data_dir)
  cache_dir = tempfile.mkdtemp()
  tff.simulation.datasets.flair.write_data(data_dir, cache_dir)
  flair_train, flair_val, flair_test = tff.simulation.datasets.flair.load_data(
      cache_dir
  )
  flair_data = collections.OrderedDict(
      train=flair_train,
      val=flair_val,
      test=flair_test,
  )
  for split, client_data in flair_data.items():
    num_clients = len(client_data.client_ids)
    logging.info('Num clients, %s split: %s', split, num_clients)


if __name__ == '__main__':
  app.run(main)
