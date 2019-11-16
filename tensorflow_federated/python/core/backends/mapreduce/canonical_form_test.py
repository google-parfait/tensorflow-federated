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

import textwrap
from absl.testing import absltest
import tensorflow.compat.v2 as tf

from tensorflow_federated.python.core.backends.mapreduce import test_utils


class CanonicalFormTest(absltest.TestCase):

  def test_summary(self):
    cf = test_utils.get_temperature_sensor_example()

    class CapturePrint(object):

      def __init__(self):
        self.summary = ''

      def __call__(self, msg):
        self.summary += msg + '\n'

    capture = CapturePrint()
    cf.summary(print_fn=capture)
    self.assertEqual(
        capture.summary,
        textwrap.dedent("""\
    initialize: ( -> <num_rounds=int32>)
    prepare   : (<num_rounds=int32> -> <max_temperature=float32>)
    work      : (<float32*,<max_temperature=float32>> -> <<is_over=bool>,<num_readings=int32>>)
    zero      : ( -> <num_total=int32,num_over=int32>)
    accumulate: (<<num_total=int32,num_over=int32>,<is_over=bool>> -> <num_total=int32,num_over=int32>)
    merge     : (<<num_total=int32,num_over=int32>,<num_total=int32,num_over=int32>> -> <num_total=int32,num_over=int32>)
    report    : (<num_total=int32,num_over=int32> -> <ratio_over_threshold=float32>)
    update    : ( -> <num_rounds=int32>)
    """))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
