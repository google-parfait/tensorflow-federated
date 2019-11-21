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
"""Utility for filtering (via class. accuracy) the Federated EMNIST dataset."""

import csv
import functools
import os.path

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.gans.experiments.emnist import emnist_data_utils

BASE_URL = 'https://storage.googleapis.com/tff-experiments-public/'
CSVS_BASE_PATH = 'gans/csvs/'


@functools.lru_cache(maxsize=1)
def get_unfiltered_client_data_for_training(batch_size):
  r"""Returns `tff.simulation.ClientData` of unfiltered Federated EMNIST data.

  The data returned will neither be filtered by user nor by example, so training
  can take place with all users and all examples for each user.

  Args:
    batch_size: Batch size of output dataset. If None, don't batch.

  Returns:
    A tff.simulation.ClientData` of real images of numbers/letters. The data has
    not been filtered.
  """
  return get_filtered_client_data_for_training(None, None, batch_size)


@functools.lru_cache(maxsize=1)
def get_filtered_by_user_client_data_for_training(invert_imagery_probability,
                                                  accuracy_threshold,
                                                  batch_size,
                                                  cache_dir=None):
  r"""Returns `tff.simulation.ClientData` of filtered Federated EMNIST data.

  Input data gets filtered on a per-user basis; users get selected via the
  `accuracy_threshold` criterion, and then training can take place with all
  examples from only the selected users.

  Args:
    invert_imagery_probability: The probability that a user\'s image data has
      pixel intensity inverted. E.g., `0p1` corresponds to 0.1, or a 10%
      probability that a user\'s data is flipped. Note that to save time in
      experiment execution, this is precomputed via the ./filter_users.py
      script, and the selection here controls which file to read from.
    accuracy_threshold: Indicates the classification threshold by which a user
      is included in the training population.  E.g., `lt0p882` means any user
      who\'s data cumulatively classifies with <0.882 accuracy would be used for
      training; `gt0p939` means any user who\'s data cumulatively classifies
      with >0.939 accuracy would be used for training. To save time in
      experiment execution, this assignment is precomputed via the
      ./filter_users.py script, and the flag selection here is to indicate which
      file to read from.
    batch_size: Batch size of output dataset. If None, don't batch.
    cache_dir: (Optional) base directory to cache the downloaded files. If None,
      caches in Keras' default cache directory.

  Returns:
    A tff.simulation.ClientData` of real images of numbers/letters. The data has
    been filtered by user classification accuracy as per the input arguments.
  """
  path_to_data = os.path.join(CSVS_BASE_PATH,
                              'inv_prob_{}'.format(invert_imagery_probability),
                              'filter_by_user',
                              'acc_{}'.format(accuracy_threshold))

  try:
    filename = 'client_ids.csv'
    path_to_read_inversions_csv = tf.keras.utils.get_file(
        fname=filename,
        cache_subdir=path_to_data,
        cache_dir=cache_dir,
        origin=os.path.join(BASE_URL, path_to_data, filename))
  except Exception:
    msg = ('A URL fetch failure was encountered when trying to retrieve '
           'filter-by-user generated csv file with invert_imagery_probability '
           '`{}` and accuracy_threshold `{}`. Please run the ./filter_users.py '
           'script to generate the missing data, and use the `cache_dir` '
           'argument to this method to specify the location of the generated '
           'data csv file.'.format(invert_imagery_probability,
                                   accuracy_threshold))
    raise ValueError(msg)

  return get_filtered_client_data_for_training(path_to_read_inversions_csv,
                                               None, batch_size)


@functools.lru_cache(maxsize=1)
def get_filtered_by_example_client_data_for_training(invert_imagery_probability,
                                                     min_num_examples,
                                                     example_class_selection,
                                                     batch_size,
                                                     cache_dir=None):
  r"""Returns `tff.simulation.ClientData` of filtered Federated EMNIST data.

  Input data gets filtered on a per-example basis. Any user meeting the
  `min_num_examples` criterion is included. The examples are limited to those
  that classified according to the `example_class_selection` criterion.

  Args:
    invert_imagery_probability: The probability that a user\'s image data has
      pixel intensity inverted. E.g., `0p1` corresponds to 0.1, or a 10%
      probability that a user\'s data is flipped. Note that to save time in
      experiment execution, this is precomputed via the ./filter_examples.py
      scripts, and the selection here controls which file to read from.
    min_num_examples: Indicates the minimum number of examples that are either
      correct or incorrect (as set by the `example_class_selection` argument) in
      a client\'s local dataset for that client to be considered as part of
      training sub-population. To save time in experiment execution, this
      assignment is precomputed via the ./filter_examples.py script, and the
      flag selection here is to indicate which file to read from.
    example_class_selection: Indicates whether to train on a client\'s correct
      or incorrect examples. To save time in experiment execution, this
      assignment is precomputed via the ./filter_examples.py script, and the
      flag selection here is to indicate which file to read from.
    batch_size: Batch size of output dataset. If None, don't batch.
    cache_dir: (Optional) base directory to cache the downloaded files. If None,
      caches in Keras' default cache directory.

  Returns:
    A `tff.simulation.ClientData` of real images of numbers/letters. The data
    has been filtered as per the input arguments (either not filtered, filtered
    by user classification accuracy, or filtered by example classification
    correctness).
  """
  path_to_data = os.path.join(CSVS_BASE_PATH,
                              'inv_prob_{}'.format(invert_imagery_probability),
                              'filter_by_example',
                              'min_num_examples_{}'.format(min_num_examples),
                              '{}'.format(example_class_selection))

  try:
    filename = 'client_ids.csv'
    path_to_read_inversions_csv = tf.keras.utils.get_file(
        fname=filename,
        cache_subdir=path_to_data,
        cache_dir=cache_dir,
        origin=os.path.join(BASE_URL, path_to_data, filename))

    filename = 'example_indices_map.csv'
    path_to_read_example_indices_csv = tf.keras.utils.get_file(
        fname=filename,
        cache_subdir=path_to_data,
        cache_dir=cache_dir,
        origin=os.path.join(BASE_URL, path_to_data, filename))
  except Exception:
    msg = ('A URL fetch failure was encountered when trying to retrieve '
           'filter-by-example generated csv files with '
           'invert_imagery_probability `{}`, min_num_examples `{}`, and '
           'example_class_selection `{}`. Please run the ./filter_examples.py '
           'script to generate the missing data, and use the `cache_dir` '
           'argument to this method to specify the location of the generated '
           'data csv files.'.format(invert_imagery_probability,
                                    min_num_examples, example_class_selection))
    raise ValueError(msg)

  return get_filtered_client_data_for_training(
      path_to_read_inversions_csv, path_to_read_example_indices_csv, batch_size)


def get_filtered_client_data_for_training(path_to_read_inversions_csv,
                                          path_to_read_example_indices_csv,
                                          batch_size):
  """Form ClientData using paths to pixel inversion, example selection data."""

  raw_client_data = emnist_data_utils.create_real_images_tff_client_data(
      'train')
  client_ids = raw_client_data.client_ids

  selected_client_ids_inversion_map = None
  client_ids_example_indices_map = None
  # If filter-by-user or filter-by-example, load the csv data into maps, and
  # update the client IDs to just the users that will be part of training.
  if path_to_read_inversions_csv is not None:
    selected_client_ids_inversion_map, client_ids_example_indices_map = (
        _get_client_ids_inversion_and_example_indices_maps(
            path_to_read_inversions_csv, path_to_read_example_indices_csv))
    client_ids = list(selected_client_ids_inversion_map.keys())

  def _get_dataset(client_id):
    """Retrieve/preprocess a tf.data.Dataset for a given client_id."""
    raw_ds = raw_client_data.create_tf_dataset_for_client(client_id)

    invert_imagery = False
    if selected_client_ids_inversion_map:
      invert_imagery = selected_client_ids_inversion_map[client_id]

    # If filter-by-example, do it here.
    if client_ids_example_indices_map:
      raw_ds = _filter_by_example(raw_ds, client_ids_example_indices_map,
                                  client_id)

    return emnist_data_utils.preprocess_img_dataset(
        raw_ds,
        invert_imagery=invert_imagery,
        include_label=False,
        batch_size=batch_size,
        shuffle=True,
        repeat=False)

  return tff.simulation.ClientData.from_clients_and_fn(client_ids, _get_dataset)


def _filter_by_example(raw_ds, client_ids_example_indices_map, client_id):
  """Form a tf.data.Dataset from the examples in the map for the client_id."""
  example_indices = client_ids_example_indices_map[client_id]
  # B/c the csv stores the list as a string, we need to do some slightly
  # klugey conversion from a string to list. (We strip off the first and
  # last characters in the string, which are [ and ], and then split on
  # commas as delimiters, to recover the original list of ints.
  example_indices = [int(s) for s in example_indices[1:-1].split(',')]

  # Get the elements (OrderedDicts) in the raw data which are at the indices
  # indicated by the list above.
  elements = []
  index = 0
  for element in raw_ds:
    if index in example_indices:
      elements.append(element)
    index += 1

  # Bind the elements (via a generator fn) into a new tf.data.Dataset.
  def _generator():
    for element in elements:
      yield element

  return tf.data.Dataset.from_generator(_generator, raw_ds.output_types,
                                        raw_ds.output_shapes)


def _get_client_ids_inversion_and_example_indices_maps(
    path_to_read_inversions_csv, path_to_read_example_indices_csv):
  """Return paths to csv files storing maps indicating the data to train on."""
  if path_to_read_inversions_csv is None:
    raise ValueError(
        'No path provided to the CSV file that stores map from client ids to '
        'image inversion data.')

  # Load (from CSV file) the specific client IDs that the GAN will train on, and
  # whether or not the images on that client are inverted.
  selected_client_ids_inversion_map = {}
  with tf.io.gfile.GFile(path_to_read_inversions_csv, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for [key, val] in csvreader:
      selected_client_ids_inversion_map[key] = (val == 'True')

  # If specified (via CSV file), the specific examples on each client ID that
  # the GAN will be trained on.
  client_ids_example_indices_map = None
  if path_to_read_example_indices_csv:
    client_ids_example_indices_map = {}
    with tf.io.gfile.GFile(path_to_read_example_indices_csv, 'r') as csvfile:
      csvreader = csv.reader(csvfile)
      for [key, val] in csvreader:
        client_ids_example_indices_map[key] = val

    set_1 = set(client_ids_example_indices_map.keys())
    set_2 = set(selected_client_ids_inversion_map.keys())
    symmetric_diff = set_1 ^ set_2
    if symmetric_diff:
      raise ValueError(
          'The CSV files at path_to_read_inversions_csv and '
          'path_to_read_example_indices_csv contain different keys.')

  return selected_client_ids_inversion_map, client_ids_example_indices_map
