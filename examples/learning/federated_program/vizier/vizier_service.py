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
"""Utilities for interacting with the Vizier service."""

import getpass

import tensorflow as tf

from google.protobuf import text_format  # pylint: disable=g-bad-import-order

from vizier.service import clients
from vizier.service import pyvizier

# The proto message type does not appear to be in the public API any longer,
# but it is necessary to be able to parse text formatted proto strings.
_StudySpec = type(pyvizier.StudyConfig().to_proto())


def create_study_config(config_path: str) -> pyvizier.StudyConfig:
  with tf.io.gfile.GFile(config_path) as f:
    proto = text_format.Parse(f.read(), _StudySpec())
  return pyvizier.StudyConfig.from_proto(proto)


def create_study(
    study_config: pyvizier.StudyConfig,
    name: str,
    owner: str = getpass.getuser(),
) -> clients.Study:
  """Creates a Vizier Study for the problem."""
  return clients.Study.from_study_config(
      config=study_config, owner=owner, study_id=name
  )
