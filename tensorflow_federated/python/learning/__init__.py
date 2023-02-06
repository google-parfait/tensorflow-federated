# Copyright 2018, The TensorFlow Federated Authors.
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
"""Libraries for building federated learning algorithms.

Currently, `tff.learning` provides a few types of functionality.

*   Algorithmic building blocks (see `tff.learning.templates`)
    for constructing federated learning algorithms. These are algorithms
    centered around the client work, server work, broadcast, or aggregation
    steps of a federated algorithms, and are intended to compose in a somewhat
    modular fashion.
*   End-to-end federated learning algorithms (such as
    `tff.learning.algorithms.build_weighted_fed_avg`) that combine broadcast,
    client work, aggregation, and server update logic into a single algorithm
    (often by composing the building blocks discussed above). This library also
    provides end-to-end algorithms for federated evaluation (see
    `tff.learning.build_federated_evaluation`).
*   Functionality supporting the development of the algorithms above. This
    includes `tff.learning.optimizers`, `tff.learning.metrics` and recommended
    aggregators, such as `tff.learning.robust_aggregator`.

The library also contains classes of models that are used for the purposes of
model training. See `tff.learning.Model` for the overall base class, and
`tff.learning.models` for related model classes.
"""

from tensorflow_federated.python.learning import algorithms
from tensorflow_federated.python.learning import framework
from tensorflow_federated.python.learning import metrics
from tensorflow_federated.python.learning import models
from tensorflow_federated.python.learning import optimizers
from tensorflow_federated.python.learning import programs
from tensorflow_federated.python.learning import reconstruction
from tensorflow_federated.python.learning import templates
from tensorflow_federated.python.learning.client_weight_lib import ClientWeighting
from tensorflow_federated.python.learning.debug_measurements import add_debug_measurements
from tensorflow_federated.python.learning.debug_measurements import add_debug_measurements_with_mixed_dtype
from tensorflow_federated.python.learning.federated_evaluation import build_federated_evaluation
from tensorflow_federated.python.learning.federated_evaluation import build_local_evaluation
from tensorflow_federated.python.learning.framework.optimizer_utils import state_with_new_model_weights
# TODO(b/267180026): Remove this once usages are migrated to
# `tff.learning.metrics`
from tensorflow_federated.python.learning.metrics.types import MetricFinalizersType
from tensorflow_federated.python.learning.model_update_aggregator import compression_aggregator
from tensorflow_federated.python.learning.model_update_aggregator import ddp_secure_aggregator
from tensorflow_federated.python.learning.model_update_aggregator import dp_aggregator
from tensorflow_federated.python.learning.model_update_aggregator import entropy_compression_aggregator
from tensorflow_federated.python.learning.model_update_aggregator import robust_aggregator
from tensorflow_federated.python.learning.model_update_aggregator import secure_aggregator
# TODO(b/267780360): Remove models/keras_utils imports once usages are migrated
# to `tff.learning.models`.
from tensorflow_federated.python.learning.models.keras_utils import federated_aggregate_keras_metric
from tensorflow_federated.python.learning.models.keras_utils import from_keras_model
from tensorflow_federated.python.learning.models.model_weights import ModelWeights
# TODO(b/259609586): Remove tff.learning.models.variable imports once all
# callsites have been updated to `tff.learning.models`.
from tensorflow_federated.python.learning.models.variable import BatchOutput
from tensorflow_federated.python.learning.models.variable import VariableModel as Model
from tensorflow_federated.python.learning.personalization_eval import build_personalization_eval
