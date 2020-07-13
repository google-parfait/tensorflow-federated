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
"""Libraries for the federated EMNIST dataset for simulation."""

import collections
import hashlib
import math
import os.path
import struct

import numpy as np
import tensorflow as tf
import tensorflow_addons.image as tfa_image

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.simulation import from_tensor_slices_client_data
from tensorflow_federated.python.simulation import hdf5_client_data
from tensorflow_federated.python.simulation import transforming_client_data


def load_data(only_digits=True, cache_dir=None):
  """Loads the Federated EMNIST dataset.

  Downloads and caches the dataset locally. If previously downloaded, tries to
  load the dataset from cache.

  This dataset is derived from the Leaf repository
  (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
  dataset, grouping examples by writer. Details about Leaf were published in
  "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.

  *Note*: This dataset does not include some additional preprocessing that
  MNIST includes, such as size-normalization and centering.
  In the Federated EMNIST data, the value of 1.0
  corresponds to the background, and 0.0 corresponds to the color of the digits
  themselves; this is the *inverse* of some MNIST representations,
  e.g. in [tensorflow_datasets]
  (https://github.com/tensorflow/datasets/blob/master/docs/datasets.md#mnist),
  where 0 corresponds to the background color, and 255 represents the color of
  the digit.

  Data set sizes:

  *only_digits=True*: 3,383 users, 10 label classes

  -   train: 341,873 examples
  -   test: 40,832 examples

  *only_digits=False*: 3,400 users, 62 label classes

  -   train: 671,585 examples
  -   test: 77,483 examples

  Rather than holding out specific users, each user's examples are split across
  _train_ and _test_ so that all users have at least one example in _train_ and
  one example in _test_. Writers that had less than 2 examples are excluded from
  the data set.

  The `tf.data.Datasets` returned by
  `tff.simulation.ClientData.create_tf_dataset_for_client` will yield
  `collections.OrderedDict` objects at each iteration, with the following keys
  and values:

    -   `'pixels'`: a `tf.Tensor` with `dtype=tf.float32` and shape [28, 28],
        containing the pixels of the handwritten digit, with values in
        the range [0.0, 1.0].
    -   `'label'`: a `tf.Tensor` with `dtype=tf.int32` and shape [1], the class
        label of the corresponding pixels. Labels [0-9] correspond to the digits
        classes, labels [10-35] correspond to the uppercase classes (e.g., label
        11 is 'B'), and labels [36-61] correspond to the lowercase classes
        (e.g., label 37 is 'b').

  Args:
    only_digits: (Optional) whether to only include examples that are from the
      digits [0-9] classes. If `False`, includes lower and upper case
      characters, for a total of 62 class labels.
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.

  Returns:
    Tuple of (train, test) where the tuple elements are
    `tff.simulation.ClientData` objects.
  """
  if only_digits:
    fileprefix = 'fed_emnist_digitsonly'
    sha256 = '55333deb8546765427c385710ca5e7301e16f4ed8b60c1dc5ae224b42bd5b14b'
  else:
    fileprefix = 'fed_emnist'
    sha256 = 'fe1ed5a502cea3a952eb105920bff8cffb32836b5173cb18a57a32c3606f3ea0'

  filename = fileprefix + '.tar.bz2'
  path = tf.keras.utils.get_file(
      filename,
      origin='https://storage.googleapis.com/tff-datasets-public/' + filename,
      file_hash=sha256,
      hash_algorithm='sha256',
      extract=True,
      archive_format='tar',
      cache_dir=cache_dir)

  dir_path = os.path.dirname(path)
  train_client_data = hdf5_client_data.HDF5ClientData(
      os.path.join(dir_path, fileprefix + '_train.h5'))
  test_client_data = hdf5_client_data.HDF5ClientData(
      os.path.join(dir_path, fileprefix + '_test.h5'))

  return train_client_data, test_client_data


def get_synthetic(num_clients=2):
  """Quickly returns a small synthetic dataset, useful for unit tests, etc.

  Each client produced has exactly 10 examples, one of each digit. The images
  are derived from a fixed set of hard-coded images, and transformed using
  `tff.simulation.datasets.emnist.get_infinite` to produce the desired number of
  clients.

  Args:
    num_clients: The number of synthetic clients to generate.

  Returns:
     A `tff.simulation.ClientData` object that matches the characteristics
     (other than size) of those provided by
     `tff.simulation.datasets.emnist.load_data`.
  """
  return get_infinite(
      # Base ClientData with one client
      from_tensor_slices_client_data.FromTensorSlicesClientData(
          {'synthetic': _get_synthetic_digits_data()}),
      num_pseudo_clients=num_clients)


def _compile_transform(angle=0,
                       shear=0,
                       scale_x=1,
                       scale_y=1,
                       translation_x=0,
                       translation_y=0):
  """Compiles affine transform parameters into single projective transform.

  The transformations are performed in the following order: rotation, shearing,
  scaling, and translation.

  Args:
    angle: The angle of counter-clockwise rotation, in degrees.
    shear: The amount of shear. Precisely, shear*x is added to the y coordinate
      after centering.
    scale_x: The amount to scale in the x-axis.
    scale_y: The amount to scale in the y-axis.
    translation_x: The number of pixels to translate in the x-axis.
    translation_y: The number of pixels to translate in the y-axis.

  Returns:
    A length 8 tensor representing the composed transform.
  """
  angle = math.radians(angle)
  size = 28

  # angles_to_projective_transforms performs rotations around center of image.
  rotation = tfa_image.transform_ops.angles_to_projective_transforms(
      angle, size, size)

  # shearing and scaling require centering and decentering.
  half = (size - 1) / 2.0
  center = tfa_image.translate_ops.translations_to_projective_transforms(
      [-half, -half])
  shear = [1., 0., 0., -shear, 1., 0., 0., 0.]
  scaling = [1. / scale_x, 0., 0., 0., 1. / scale_y, 0., 0., 0.]
  decenter = tfa_image.translate_ops.translations_to_projective_transforms(
      [half, half])

  translation = tfa_image.translate_ops.translations_to_projective_transforms(
      [translation_x, translation_y])
  return tfa_image.transform_ops.compose_transforms(
      transforms=[rotation, center, shear, scaling, decenter, translation])


def _make_transform_fn(raw_client_id, index):
  """Generates a pseudorandom affine transform based on the client_id and index.

  If the index is 0, `None` is returned so no transform is applied by the
  transforming_client_data.

  Args:
    raw_client_id: The raw client_id.
    index: The index of the pseudo-client.

  Returns:
    The transformed data.
  """
  if index == 0:
    return None

  py_typecheck.check_type(raw_client_id, str)
  # To be python2 compatible, we need to use struct.unpack() to convert bytes to
  # int. (In python3, the int.from_bytes() method could be used instead.)
  _, _, _, stable_hash_of_client_id = struct.unpack(
      '>IIII',
      hashlib.md5(raw_client_id.encode()).digest())
  np.random.seed((stable_hash_of_client_id + index) % (2**32))

  def random_scale(min_val):
    b = math.log(min_val)
    return math.exp(np.random.uniform(b, -b))

  transform = _compile_transform(
      angle=np.random.uniform(-20, 20),
      shear=np.random.uniform(-0.2, 0.2),
      scale_x=random_scale(0.8),
      scale_y=random_scale(0.8),
      translation_x=np.random.uniform(-5, 5),
      translation_y=np.random.uniform(-5, 5))

  def _transform_fn(data):
    """Applies a random transform to the pixels."""
    # EMNIST background is 1.0 but tfa_image.transform assumes 0.0, so invert.
    pixels = 1.0 - data['pixels']

    pixels = tfa_image.transform(pixels, transform, 'BILINEAR')

    # num_bits=9 actually yields 256 unique values.
    pixels = tf.quantization.quantize_and_dequantize(
        pixels, 0.0, 1.0, num_bits=9, range_given=True)

    data['pixels'] = 1.0 - pixels
    return data

  return _transform_fn


def get_infinite(emnist_client_data, num_pseudo_clients):
  """Converts a Federated EMNIST dataset into an Infinite Federated EMNIST set.

  Infinite Federated EMNIST expands each writer from the EMNIST dataset into
  some number of pseudo-clients each of whose characters are the same but apply
  a fixed random affine transformation to the original user's characters. The
  distribution over affine transformation is approximately equivalent to the one
  described at https://www.cs.toronto.edu/~tijmen/affNIST/. It applies the
  following transformations in this order:

    1. A random rotation chosen uniformly between -20 and 20 degrees.
    2. A random shearing adding between -0.2 to 0.2 of the x coordinate to the
       y coordinate (after centering).
    3. A random scaling between 0.8 and 1.25 (sampled log uniformly).
    4. A random translation between -5 and 5 pixels in both the x and y axes.

  Args:
    emnist_client_data: The `tff.simulation.ClientData` to convert.
    num_pseudo_clients: How many pseudo-clients to generate for each real
      client. Each pseudo-client is formed by applying a given random affine
      transformation to the characters written by a given real user. The first
      pseudo-client for a given user applies the identity transformation, so the
      original users are always included.

  Returns:
    An expanded `tff.simulation.ClientData`.
  """
  num_client_ids = len(emnist_client_data.client_ids)

  return transforming_client_data.TransformingClientData(
      raw_client_data=emnist_client_data,
      make_transform_fn=_make_transform_fn,
      num_transformed_clients=(num_client_ids * num_pseudo_clients))


def _get_synthetic_digits_data():
  """Returns a dictionary suitable for `tf.data.Dataset.from_tensor_slices`.

  Returns:
    A dictionary that matches the structure of the data produced by
    `tff.simulation.datasets.emnist.load_data`, with keys `pixels` and `label`.
  """
  data = np.array(_SYNTHETIC_DIGITS_DATA)
  img_list = []
  for img_array in data:
    img_array = img_array.astype(np.float32) / 9.0
    img_list.append(img_array)
  assert len(img_list) == 10
  return collections.OrderedDict([('pixels', img_list),
                                  ('label', list(range(10)))])


# pyformat: disable
# pylint: disable=bad-continuation,bad-whitespace
_SYNTHETIC_DIGITS_DATA = [
   [[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]],

   [[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,7,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,2,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,2,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,2,0,0,0,2,4,4,4,7,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,0,0,0,0,0,0,0,0,4,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,4,4,4,4,4,4,4,4,7,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]],

   [[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,7,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,7,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,2,4,2,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,0,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,4,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]],

   [[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,7,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,7,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]],

   [[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,7,4,4,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,7,4,2,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,2,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,2,4,2,0,0,0,2,4,7,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,0,0,0,0,0,0,0,0,4,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,4,4,2,0,0,0,2,4,7,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,7,4,4,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]],

   [[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,4,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,0,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,2,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,2,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,7,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,7,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,4,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,2,4,4,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]],

   [[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,7,4,4,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,2,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]],

   [[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,4,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,0,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,2,4,2,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,7,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,7,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]],

   [[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]],

   [[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,2,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,4,4,7,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,4,0,0,0,4,9,4,0,0,0,4,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,7,4,2,0,2,4,2,0,2,4,7,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,4,0,0,0,0,0,4,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,7,4,4,4,4,4,7,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
    [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]]
]
