## Copyright 2022, The TensorFlow Federated Authors.
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
"""Python type annotations aliases."""

from typing import OrderedDict, Union

import numpy as np
import tensorflow as tf

# TODO(b/243388116): add a pyttype alias for Datasets/sequences of tensors,
# and potentially a type that is Union[TensorLike, TensorSequenceLike].

# A type denoting tensor-like objects.
#
# This is useful for talking about non-composite types that are compatible with
# `tf.convert_to_tensor`, parameters of `tf.function`, and similar usages.
TensorLike = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor, tf.Variable,
                   np.ndarray, np.number, float, int, str, bytes]

# A nested structure of TensorLike objects.
#
# Intended to capture types that are compatible with `tf.nest` APIs.
TensorStructure = Union[list[Union[TensorLike, 'TensorStructure']],
                        tuple[Union[TensorLike, 'TensorStructure'], ...],
                        OrderedDict[str, Union[TensorLike, 'TensorStructure']]]

# A variant type covering any type of tensor spec from TensorFlow.
TensorSpecVariant = Union[tf.TensorSpec, tf.SparseTensorSpec,
                          tf.RaggedTensorSpec]

# A potentially nested structure of TensorSpecVariant objects.
TensorSpecStructure = Union[list[Union[TensorSpecVariant,
                                       'TensorSpecStructure']],
                            tuple[Union[TensorSpecVariant,
                                        'TensorSpecStructure'], ...],
                            OrderedDict[str, Union[TensorSpecVariant,
                                                   'TensorSpecStructure']]]
