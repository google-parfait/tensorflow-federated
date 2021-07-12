# Copyright 2021, The TensorFlow Federated Authors.
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

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import vision_datasets_utils as utils


class VisionDatasetsUtilsTest(tf.test.TestCase):

  def test_create_example(self):
    image_bytes_1 = b'somebytes'
    image_bytes_2 = b'someotherbytes'
    created_examples = [
        utils.create_example(image_bytes_1, 0),
        utils.create_example(image_bytes_2, 1)
    ]

    expected_dataset = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    utils.KEY_IMAGE_BYTES:
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[image_bytes_1])),
                    utils.KEY_CLASS:
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[0])),
                })),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    utils.KEY_IMAGE_BYTES:
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[image_bytes_2])),
                    utils.KEY_CLASS:
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[1])),
                })),
    ]

    self.assertSequenceEqual(created_examples, expected_dataset)


if __name__ == '__main__':
  tf.test.main()
