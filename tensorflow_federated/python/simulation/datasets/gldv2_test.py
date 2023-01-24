# Copyright 2020, The TensorFlow Federated Authors.
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

import collections

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import gldv2
from tensorflow_federated.python.simulation.datasets import vision_datasets_utils


class GLDV2Test(tf.test.TestCase):

  def test_create_dataset_from_mapping(self):
    tmp_dir = self.create_tempdir(name='images')
    images = {'image_1': b'somebytes', 'image_2': b'someotherbytes'}
    for image_name, content in images.items():
      tmp_dir.create_file(file_path=image_name + '.jpg', content=content)

    mapping = [
        {'image_id': 'image_1', 'class': '0'},
        {'image_id': 'image_2', 'class': '12'},
    ]

    expected_features = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    vision_datasets_utils.KEY_IMAGE_BYTES: tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b'somebytes'])
                    ),
                    vision_datasets_utils.KEY_CLASS: tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[0])
                    ),
                }
            )
        ),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    vision_datasets_utils.KEY_IMAGE_BYTES: tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b'someotherbytes'])
                    ),
                    vision_datasets_utils.KEY_CLASS: tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[12])
                    ),
                }
            )
        ),
    ]

    features = gldv2._create_dataset_with_mapping(tmp_dir.full_path, mapping)
    self.assertSequenceEqual(features, expected_features)

  def test_get_synthetic(self):
    client_data = gldv2.get_synthetic()
    self.assertLen(client_data.client_ids, 1)
    expected_element_type = collections.OrderedDict([
        (
            'image/decoded',
            tf.TensorSpec(shape=(600, 800, 3), dtype=tf.uint8),
        ),
        ('class', tf.TensorSpec(shape=(1,), dtype=tf.int64)),
    ])
    self.assertEqual(client_data.element_type_structure, expected_element_type)
    data = client_data.create_tf_dataset_for_client(client_data.client_ids[0])
    images = [element['image/decoded'] for element in data]
    self.assertLen(images, 3)
    labels = [element['class'] for element in data]
    self.assertEqual(labels, [[0], [1], [2]])


if __name__ == '__main__':
  tf.test.main()
