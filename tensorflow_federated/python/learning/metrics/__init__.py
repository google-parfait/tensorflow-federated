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
"""Libraries for working with metrics in federated learning algorithms."""

from tensorflow_federated.python.learning.metrics.aggregation_factory import create_default_secure_sum_quantization_ranges
from tensorflow_federated.python.learning.metrics.aggregation_factory import SecureSumFactory
from tensorflow_federated.python.learning.metrics.aggregation_factory import SumThenFinalizeFactory
from tensorflow_federated.python.learning.metrics.aggregator import secure_sum_then_finalize
from tensorflow_federated.python.learning.metrics.aggregator import sum_then_finalize
from tensorflow_federated.python.learning.metrics.counters import NumBatchesCounter
from tensorflow_federated.python.learning.metrics.counters import NumExamplesCounter
from tensorflow_federated.python.learning.metrics.keras_finalizer import create_keras_metric_finalizer
from tensorflow_federated.python.learning.metrics.keras_utils import create_functional_metric_fns
from tensorflow_federated.python.learning.metrics.types import FunctionalMetricFinalizersType
from tensorflow_federated.python.learning.metrics.types import MetricFinalizersType
