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
"""Experimental federated learning components for JAX."""

import collections

import jax
import numpy as np

from tensorflow_federated.experimental.python.core.api import computations as experimental_computations
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import iterative_process

# TODO(b/175888145): Evolve this to reach parity with TensorFlow-specific helper
# and eventually unify the two.


def build_jax_federated_averaging_process(batch_type, model_type, loss_fn,
                                          step_size):
  """Constructs an iterative process that implements simple federated averaging.

  Args:
    batch_type: An instance of `tff.Type` that represents the type of a single
      batch of data to use for training. This type should be constructed with
      standard Python containers (such as `collections.OrderedDict`) of the sort
      that are expected as parameters to `loss_fn`.
    model_type: An instance of `tff.Type` that represents the type of the model.
      Similarly to `batch_size`, this type should be constructed with standard
      Python containers (such as `collections.OrderedDict`) of the sort that are
      expected as parameters to `loss_fn`.
    loss_fn: A loss function for the model. Must be a Python function that takes
      two parameters, one of them being the model, and the other being a single
      batch of data (with types matching `batch_type` and `model_type`).
    step_size: The step size to use during training (an `np.float32`).

  Returns:
    An instance of `tff.templates.IterativeProcess` that implements federated
    training in JAX.
  """
  batch_type = computation_types.to_type(batch_type)
  model_type = computation_types.to_type(model_type)

  py_typecheck.check_type(batch_type, computation_types.Type)
  py_typecheck.check_type(model_type, computation_types.Type)
  py_typecheck.check_callable(loss_fn)
  py_typecheck.check_type(step_size, np.float)

  def _tensor_zeros(tensor_type):
    return jax.numpy.zeros(
        tensor_type.shape.dims, dtype=tensor_type.dtype.as_numpy_dtype)

  @experimental_computations.jax_computation
  def _create_zero_model():
    model_zeros = structure.map_structure(_tensor_zeros, model_type)
    return type_conversions.type_to_py_container(model_zeros, model_type)

  @computations.federated_computation
  def _create_zero_model_on_server():
    return intrinsics.federated_eval(_create_zero_model, placements.SERVER)

  def _apply_update(model_param, param_delta):
    return model_param - step_size * param_delta

  @experimental_computations.jax_computation(model_type, batch_type)
  def _train_on_one_batch(model, batch):
    params = structure.flatten(structure.from_container(model, recursive=True))
    grads = structure.flatten(
        structure.from_container(jax.api.grad(loss_fn)(model, batch)))
    updated_params = [_apply_update(x, y) for (x, y) in zip(params, grads)]
    trained_model = structure.pack_sequence_as(model_type, updated_params)
    return type_conversions.type_to_py_container(trained_model, model_type)

  local_dataset_type = computation_types.SequenceType(batch_type)

  @computations.federated_computation(model_type, local_dataset_type)
  def _train_on_one_client(model, batches):
    return intrinsics.sequence_reduce(batches, model, _train_on_one_batch)

  @computations.federated_computation(
      computation_types.FederatedType(model_type, placements.SERVER),
      computation_types.FederatedType(local_dataset_type, placements.CLIENTS))
  def _train_one_round(model, federated_data):
    locally_trained_models = intrinsics.federated_map(
        _train_on_one_client,
        collections.OrderedDict([('model',
                                  intrinsics.federated_broadcast(model)),
                                 ('batches', federated_data)]))
    return intrinsics.federated_mean(locally_trained_models)

  return iterative_process.IterativeProcess(
      initialize_fn=_create_zero_model_on_server, next_fn=_train_one_round)
