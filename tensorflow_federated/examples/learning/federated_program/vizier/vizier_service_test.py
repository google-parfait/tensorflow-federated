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

import os.path
from unittest import mock

from absl.testing import absltest

from tensorflow_federated.examples.learning.federated_program.vizier import vizier_service

from vizier.service import clients
from vizier.service import pyvizier

_STUDY_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    'study_spec.textproto',
)


class VizierServiceTest(absltest.TestCase):

  def test_create_study_config_returns_result(self):
    study_config = vizier_service.create_study_config(_STUDY_CONFIG_PATH)
    self.assertIsInstance(study_config, pyvizier.StudyConfig)

  def test_create_study_returns_result(self):
    study_config = vizier_service.create_study_config(_STUDY_CONFIG_PATH)

    with mock.patch.object(
        clients, 'Study', autospec=True, spec_set=True
    ) as mock_study:
      vizier_service.create_study(
          study_config=study_config, name='test', owner='owner'
      )
      mock_study.from_study_config.assert_called_once_with(
          config=study_config, owner='owner', study_id='test'
      )


if __name__ == '__main__':
  absltest.main()
