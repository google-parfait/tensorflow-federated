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
"""Libraries for the federated CIFAR-100 dataset for simulation."""

import collections

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import download
from tensorflow_federated.python.simulation.datasets import from_tensor_slices_client_data
from tensorflow_federated.python.simulation.datasets import sql_client_data


def _add_proto_parsing(dataset: tf.data.Dataset) -> tf.data.Dataset:
  """Add parsing of the tf.Example proto to the dataset pipeline."""

  def parse_proto(tensor_proto):
    parse_spec = {
        'coarse_label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'image': tf.io.FixedLenFeature(shape=(32, 32, 3), dtype=tf.int64),
        'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    }
    parsed_features = tf.io.parse_example(tensor_proto, parse_spec)
    return collections.OrderedDict(
        coarse_label=parsed_features['coarse_label'],
        image=tf.cast(parsed_features['image'], tf.uint8),
        label=parsed_features['label'],
    )

  return dataset.map(parse_proto, num_parallel_calls=tf.data.AUTOTUNE)


def load_data(cache_dir=None):
  """Loads a federated version of the CIFAR-100 dataset.

  The dataset is downloaded and cached locally. If previously downloaded, it
  tries to load the dataset from cache.

  The dataset is derived from the [CIFAR-100
  dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The training and
  testing examples are partitioned across 500 and 100 clients (respectively).
  No clients share any data samples, so it is a true partition of CIFAR-100. The
  train clients have string client IDs in the range [0-499], while the test
  clients have string client IDs in the range [0-99]. The train clients form a
  true partition of the CIFAR-100 training split, while the test clients form a
  true partition of the CIFAR-100 testing split.

  The data partitioning is done using a hierarchical Latent Dirichlet Allocation
  (LDA) process, referred to as the [Pachinko Allocation Method]
  (https://people.cs.umass.edu/~mccallum/papers/pam-icml06.pdf) (PAM).
  This method uses a two-stage LDA process, where each client has an associated
  multinomial distribution over the coarse labels of CIFAR-100, and a
  coarse-to-fine label multinomial distribution for that coarse label over the
  labels under that coarse label. The coarse label multinomial is drawn from a
  symmetric Dirichlet with parameter 0.1, and each coarse-to-fine multinomial
  distribution is drawn from a symmetric Dirichlet with parameter 10. Each
  client has 100 samples. To generate a sample for the client, we first select
  a coarse label by drawing from the coarse label multinomial distribution, and
  then draw a fine label using the coarse-to-fine multinomial distribution. We
  then randomly draw a sample from CIFAR-100 with that label (without
  replacement). If this exhausts the set of samples with this label, we
  remove the label from the coarse-to-fine multinomial and renormalize the
  multinomial distribution.

  Data set sizes:
  -   train: 50,000 examples
  -   test: 10,000 examples

  The `tf.data.Datasets` returned by
  `tff.simulation.datasets.ClientData.create_tf_dataset_for_client` will yield
  `collections.OrderedDict` objects at each iteration, with the following keys
  and values, in lexicographic order by key:

    -   `'coarse_label'`: a `tf.Tensor` with `dtype=tf.int64` and shape [1] that
        corresponds to the coarse label of the associated image. Labels are
        in the range [0-19].
    -   `'image'`: a `tf.Tensor` with `dtype=tf.uint8` and shape [32, 32, 3],
        containing the red/blue/green pixels of the image. Each pixel is a value
        in the range [0, 255].
    -   `'label'`: a `tf.Tensor` with `dtype=tf.int64` and shape [1], the class
        label of the corresponding image. Labels are in the range [0-99].

  Args:
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.

  Returns:
    Tuple of (train, test) where the tuple elements are
    `tff.simulation.datasets.ClientData` objects.
  """
  database_path = download.get_compressed_file(
      origin='https://storage.googleapis.com/tff-datasets-public/cifar100.sqlite.lzma',
      cache_dir=cache_dir,
  )
  train_client_data = sql_client_data.SqlClientData(
      database_path, 'train'
  ).preprocess(_add_proto_parsing)
  test_client_data = sql_client_data.SqlClientData(
      database_path, 'test'
  ).preprocess(_add_proto_parsing)
  return train_client_data, test_client_data


def get_synthetic():
  """Returns a small synthetic dataset for testing.

  The two clients produced have exactly 5 examples apiece. The images and
  labels are derived from a fixed set of hard-coded images.

  Returns:
     A `tff.simulation.datasets.ClientData` object that matches the
     characteristics (other than size) of those provided by
     `tff.simulation.datasets.cifar100.load_data`.
  """
  return from_tensor_slices_client_data.TestClientData({
      'synthetic1': _get_synthetic_digits_data(),
      'synthetic2': _get_synthetic_digits_data(),
  })


def _get_synthetic_digits_data():
  """Returns a dictionary suitable for `tf.data.Dataset.from_tensor_slices`.

  Returns:
    A dictionary that matches the structure of the data produced by
    `tff.simulation.datasets.cifar100.load_data`, with keys (in lexicographic
    order) `coarse_label`, `image` and `label`.
  """
  data = _SYNTHETIC_IMAGE_DATA
  images = []
  for img_array in data:
    reshaped_image = tf.image.resize(img_array, (32, 32))
    images.append(tf.cast(reshaped_image, dtype=tf.uint8))

  images = tf.stack(images, axis=0)
  coarse_labels = tf.constant([4, 4, 4, 8, 10], dtype=tf.int64)
  labels = tf.constant([0, 51, 51, 88, 71], dtype=tf.int64)

  data = collections.OrderedDict(
      coarse_label=coarse_labels, image=images, label=labels
  )
  return tf.nest.map_structure(lambda x: x.numpy(), data)


# This consists of 5 CIFAR-like that have been downsampled to images of shape
# (8, 8, 3), and have been converted to float values. To re-convert to
# CIFAR-like images, we upsample to (32, 32, 3) and recast to `tf.uint8`.
_SYNTHETIC_IMAGE_DATA = [
    [
        [
            [159, 149.5, 109],
            [117.75, 123, 116.75],
            [132.5, 138, 119.5],
            [159, 171.75, 107.25],
            [131.75, 143.25, 100.25],
            [142.75, 149.5, 133],
            [137, 140, 146],
            [134.5, 138.25, 143],
        ],
        [
            [125.75, 129, 136],
            [125.25, 138.25, 67.25],
            [164.5, 186.75, 73.75],
            [159.25, 185.25, 58.5],
            [123.5, 146, 49.5],
            [139.75, 170, 37.5],
            [174, 200.25, 68.25],
            [143, 147.75, 149.5],
        ],
        [
            [137, 140.25, 140.75],
            [129.25, 151, 24.5],
            [163.5, 183.75, 42.75],
            [180.75, 204, 60.5],
            [187.5, 211.5, 85.75],
            [183.5, 210.75, 93.5],
            [187.5, 206.5, 70.75],
            [158.25, 165, 113.5],
        ],
        [
            [104, 109.75, 87],
            [113.5, 128.75, 4.75],
            [154.25, 173, 27.5],
            [171, 195.5, 48.75],
            [198, 215.5, 106.25],
            [212.25, 226.75, 143.75],
            [189.5, 206.25, 69.5],
            [163.25, 174.25, 85],
        ],
        [
            [47.5, 50.75, 43.5],
            [102, 114.75, 3],
            [134.75, 154, 18.25],
            [155, 176.25, 32],
            [176.25, 197.75, 57.25],
            [177, 201.25, 61.5],
            [182.5, 203.25, 58.5],
            [150.75, 158.5, 106.25],
        ],
        [
            [15, 15.75, 23.5],
            [67.5, 77.5, 5.5],
            [112, 127.75, 3.25],
            [146.25, 163.5, 26],
            [157.25, 178.75, 36.75],
            [166.25, 188, 46.5],
            [169, 188, 46.5],
            [132.5, 137.75, 129],
        ],
        [
            [13.25, 15.75, 21.5],
            [66.5, 72, 34],
            [84.75, 97, 5.75],
            [128.5, 146.25, 17.5],
            [135.75, 156.75, 22.75],
            [143.75, 165, 31.75],
            [134, 144.75, 72.25],
            [119.25, 121.75, 124.25],
        ],
        [
            [26.5, 29.5, 34.5],
            [11.75, 11, 16],
            [60, 64, 31.5],
            [85.75, 93.5, 38.5],
            [84.25, 93.5, 36],
            [125.5, 130, 101.5],
            [123.75, 124.25, 127.75],
            [113.75, 115.75, 116],
        ],
    ],
    [
        [
            [97.5, 117.25, 130.25],
            [85.25, 78.5, 88.75],
            [55.25, 49.75, 57.75],
            [42.75, 52, 48.25],
            [125.5, 138, 136.75],
            [252.5, 252.75, 252.75],
            [254.5, 254.5, 254.5],
            [254.5, 254.5, 254.5],
        ],
        [
            [98.25, 114, 125.5],
            [91.75, 82, 92.5],
            [95.75, 84.75, 94.25],
            [89.25, 75, 86],
            [80.75, 76.5, 86.5],
            [158.25, 165.25, 160],
            [255, 255, 255],
            [255, 255, 255],
        ],
        [
            [69.25, 74.5, 85.5],
            [91.5, 79.25, 89.75],
            [112, 96.5, 102.75],
            [102.75, 89, 99.75],
            [165, 153.5, 159.25],
            [240, 240.25, 240.75],
            [255, 255, 255],
            [255, 255, 255],
        ],
        [
            [52.25, 86.5, 35.75],
            [37.5, 43.25, 38.75],
            [10, 9.5, 19.5],
            [8.25, 8.75, 19.25],
            [11.25, 11.5, 23.75],
            [108.75, 110.5, 115],
            [252.75, 253.5, 253.5],
            [252.5, 253, 252.5],
        ],
        [
            [72.75, 107, 70.5],
            [47, 90.25, 39.5],
            [38, 62.25, 34],
            [33.25, 52.25, 34.5],
            [94.25, 91.75, 88],
            [124.5, 123, 123.5],
            [133.75, 147.75, 144],
            [63.25, 76.25, 65.75],
        ],
        [
            [40.5, 74.75, 36.75],
            [73.75, 109, 70],
            [60.5, 97.75, 50.5],
            [54.25, 97.25, 38.5],
            [48.75, 77, 48.75],
            [65.5, 95.25, 63],
            [105.75, 131.5, 95.25],
            [83.75, 124.5, 58],
        ],
        [
            [44.5, 90.5, 33.75],
            [59, 99, 54.5],
            [51.5, 103.75, 32.5],
            [63.5, 111, 39.5],
            [67.5, 109, 42.75],
            [53.25, 92, 44],
            [39.75, 80, 28.75],
            [52.5, 91, 53.25],
        ],
        [
            [32.25, 69.5, 23.25],
            [42, 75, 38.75],
            [65.5, 106.25, 57.25],
            [58, 97.5, 47.5],
            [52.5, 77.25, 45],
            [61.25, 97, 44.25],
            [62.75, 100.75, 55.25],
            [46, 92.5, 30.75],
        ],
    ],
    [
        [
            [60.25, 49.75, 36],
            [72, 78.5, 35.5],
            [134.5, 125.75, 91],
            [214.5, 196.75, 163.5],
            [207.75, 174, 133.5],
            [151.75, 125.75, 85.5],
            [75.5, 91, 28],
            [54.25, 67, 25],
        ],
        [
            [54.75, 58.5, 30.5],
            [77.75, 64.25, 41.5],
            [114, 104.5, 67.25],
            [203.25, 183.5, 144],
            [229.5, 173.25, 113.75],
            [240, 182.5, 109],
            [146, 110.5, 61.75],
            [75.5, 75.25, 43],
        ],
        [
            [88.25, 99, 35.5],
            [168, 215, 68],
            [158, 194.75, 61],
            [160, 137.25, 99.25],
            [186.75, 138, 91],
            [214.5, 168.75, 110],
            [82.5, 65.25, 36.25],
            [75.5, 86.25, 36.25],
        ],
        [
            [120.5, 152.25, 53],
            [112.75, 150.5, 58],
            [179.75, 205, 107.75],
            [141, 110.5, 76],
            [164.25, 117.5, 78],
            [152.5, 110.5, 72],
            [128.5, 102, 64.25],
            [101, 107, 50.5],
        ],
        [
            [118.25, 144.25, 56],
            [115.75, 130.25, 55],
            [76, 71.25, 32.5],
            [231, 154.25, 87.25],
            [185.25, 127, 73.5],
            [140, 101.5, 61.75],
            [131.75, 111.75, 61],
            [83.5, 87, 44.5],
        ],
        [
            [96.25, 103, 56.5],
            [153, 155.5, 91],
            [92.25, 69.5, 32.75],
            [236, 169.75, 109.5],
            [178.5, 138.25, 71.75],
            [122.25, 101, 59.5],
            [139.75, 114.5, 73.5],
            [96.5, 79, 51.5],
        ],
        [
            [159.5, 210, 90],
            [194.75, 244.25, 101.75],
            [130, 101.75, 74.75],
            [226.25, 148, 81],
            [154, 183.5, 86.75],
            [120.25, 98, 62],
            [119.5, 102.75, 71],
            [151, 136.75, 116],
        ],
        [
            [111.5, 145.75, 66],
            [193, 245.25, 121.25],
            [118.75, 111, 65.75],
            [149.75, 166, 91.25],
            [161, 137.5, 95.25],
            [129.25, 123.5, 73.75],
            [95.5, 89.5, 60.25],
            [140, 137.5, 110.25],
        ],
    ],
    [
        [
            [62.75, 95.5, 63.5],
            [37.75, 60.75, 32.25],
            [78, 91.75, 75.5],
            [48.25, 59.5, 47],
            [56.25, 55, 41],
            [54.25, 61, 43],
            [33.25, 53, 45],
            [33.25, 40.75, 40.5],
        ],
        [
            [48.25, 65.25, 55.25],
            [40.25, 55, 42.75],
            [87, 76.5, 66.75],
            [47.75, 73, 49],
            [69.5, 66.5, 60.75],
            [144, 144, 134.75],
            [32.75, 43.75, 34.25],
            [39.5, 48.25, 45],
        ],
        [
            [47.75, 59.5, 52.5],
            [48, 53.75, 43.75],
            [48.25, 48.5, 45.75],
            [45, 52.5, 40.5],
            [101.5, 82.25, 61],
            [137.25, 127.75, 105.5],
            [90.5, 93, 81.75],
            [34.75, 46, 41.25],
        ],
        [
            [87.5, 102.25, 99.25],
            [84.75, 94.5, 86.75],
            [102.5, 107.5, 110.5],
            [128, 116, 96],
            [102.5, 82.25, 60],
            [124.25, 119, 109.5],
            [141.25, 138.75, 110],
            [50.75, 65.75, 59],
        ],
        [
            [49.25, 60.5, 54.25],
            [78.75, 82.75, 78.25],
            [99, 108.25, 117.25],
            [77.5, 66.5, 50.25],
            [124, 108, 88],
            [130.25, 138.25, 140],
            [158, 145.75, 120.25],
            [103, 116.75, 114],
        ],
        [
            [101.25, 126.25, 123.25],
            [107, 119.5, 124.5],
            [138.75, 143.5, 143],
            [108, 110, 101.5],
            [70.5, 70.25, 61.25],
            [142, 124, 88.75],
            [134.5, 138.5, 116.75],
            [155.25, 181, 181.5],
        ],
        [
            [175.5, 199.5, 195.75],
            [206.75, 221, 214.75],
            [168.5, 183.75, 170],
            [166.5, 186.25, 176.5],
            [179.5, 189, 185.25],
            [141.5, 144.75, 133.75],
            [169.25, 190.5, 185],
            [165.25, 156.25, 143.25],
        ],
        [
            [144.25, 165, 147.5],
            [163.25, 183, 163],
            [162, 162, 132.75],
            [165.5, 151, 127],
            [153.5, 138.25, 116.75],
            [185.75, 185.5, 166.5],
            [148.75, 137, 117.5],
            [149.5, 121, 105],
        ],
    ],
    [
        [
            [95.5, 103.25, 121],
            [127.5, 134.75, 151.75],
            [133, 141.75, 161.5],
            [97.75, 115.5, 149.25],
            [87, 107.75, 148],
            [95.75, 112.75, 149],
            [88, 107.25, 142.5],
            [79.5, 97.5, 128.5],
        ],
        [
            [68, 88, 120.75],
            [75.25, 100.75, 136.25],
            [85.5, 109.5, 148.25],
            [96.5, 119.25, 159],
            [143.25, 157, 180],
            [209.75, 210.75, 217.75],
            [177.75, 179, 186.25],
            [125.75, 131, 144.75],
        ],
        [
            [73.25, 77, 88.75],
            [103.25, 102.75, 105.75],
            [113.75, 112.5, 116.5],
            [133, 129.75, 127.75],
            [147.5, 142.5, 137],
            [219.25, 219.5, 218.75],
            [95.25, 92.25, 95.25],
            [56.5, 53.75, 59.5],
        ],
        [
            [119.75, 112.25, 95],
            [125.5, 110.25, 91],
            [109.5, 96.25, 83.5],
            [146.5, 125.25, 101.5],
            [171, 147, 119],
            [185, 164.25, 133.75],
            [171, 149, 118],
            [136, 122.5, 95.75],
        ],
        [
            [52, 49.25, 45.75],
            [48, 46.75, 47.25],
            [59.5, 55, 54.5],
            [83.5, 70.5, 61.75],
            [104.25, 83.25, 66.75],
            [124.25, 98.75, 75.75],
            [79, 65.25, 56],
            [60, 51, 47.5],
        ],
        [
            [24, 23.5, 22.5],
            [25.25, 24.75, 25],
            [35.25, 33.5, 31],
            [48, 41.25, 34.5],
            [156, 118.5, 74.25],
            [236.25, 226, 197.25],
            [133.75, 96.25, 62.75],
            [39.5, 32.75, 27.5],
        ],
        [
            [7.75, 7.75, 9.75],
            [13, 12, 12.5],
            [17.75, 17.75, 15.75],
            [34.75, 27.75, 22.25],
            [111.75, 76.5, 43.5],
            [133.25, 92.75, 52],
            [67.25, 45.5, 32.5],
            [34.25, 27.25, 23.25],
        ],
        [
            [4, 4, 4],
            [9, 7, 6.5],
            [11.5, 10.5, 8.75],
            [18, 14.25, 11],
            [35, 23, 14.25],
            [60.5, 36, 19],
            [30.5, 21, 12.5],
            [14.5, 11.75, 6.75],
        ],
    ],
]
