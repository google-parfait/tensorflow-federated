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
"""Libraries providing implementations of federated learning algorithms."""

from tensorflow_federated.python.learning.algorithms.fed_avg import build_unweighted_fed_avg
from tensorflow_federated.python.learning.algorithms.fed_avg import build_weighted_fed_avg
from tensorflow_federated.python.learning.algorithms.fed_avg_with_optimizer_schedule import build_weighted_fed_avg_with_optimizer_schedule
from tensorflow_federated.python.learning.algorithms.fed_eval import build_fed_eval
from tensorflow_federated.python.learning.algorithms.fed_prox import build_unweighted_fed_prox
from tensorflow_federated.python.learning.algorithms.fed_prox import build_weighted_fed_prox
from tensorflow_federated.python.learning.algorithms.fed_sgd import build_fed_sgd
from tensorflow_federated.python.learning.algorithms.kmeans_clustering import build_fed_kmeans
from tensorflow_federated.python.learning.algorithms.mime import build_mime_lite_with_optimizer_schedule
from tensorflow_federated.python.learning.algorithms.mime import build_unweighted_mime_lite
from tensorflow_federated.python.learning.algorithms.mime import build_weighted_mime_lite
