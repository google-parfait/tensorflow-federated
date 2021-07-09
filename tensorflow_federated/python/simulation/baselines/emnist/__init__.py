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
"""Libraries for constructing baseline tasks for the EMNIST dataset."""

from tensorflow_federated.python.simulation.baselines.emnist.autoencoder_tasks import create_autoencoder_task
from tensorflow_federated.python.simulation.baselines.emnist.char_recognition_tasks import CharacterRecognitionModel
from tensorflow_federated.python.simulation.baselines.emnist.char_recognition_tasks import create_character_recognition_task
