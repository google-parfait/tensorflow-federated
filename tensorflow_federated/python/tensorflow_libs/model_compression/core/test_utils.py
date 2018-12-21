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
"""Testing utilities for the `model_compression` module.

This file contains:
* Base test class for testing implementations of the `EncodingStageInterface`.
* Example implementations of the `EncodingStageInterface`. These example
implementations are used to test the base test class, and the `Encoder` class.
* Other utilities useful for testing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

# Dependency imports
from absl.testing import parameterized
import numpy as np
import six
import tensorflow as tf

# TODO(b/118783928) Fix BUILD target visibility.
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import nest

from tensorflow_federated.python.tensorflow_libs.model_compression.core import encoding_stage
from tensorflow_federated.python.tensorflow_libs.model_compression.core import utils

# Named tuple containing the values summarizing the results for a single
# evaluation of an encoding stage.
TestData = collections.namedtuple(
    'TestData',
    [
        'x',  # The input provided to encoding.
        'encoded_x',  # A dictionary of values representing the encoded input x.
        'decoded_x',  # Decoded value. Has the same shape as x.
    ])


# This metaclass enables adding abc.ABCMeta metaclass to a class inheriting from
# parameterized.TestCase.
class ParameterizedABCMeta(abc.ABCMeta, parameterized.TestGeneratorMetaclass):
  pass


@six.add_metaclass(ParameterizedABCMeta)
class BaseEncodingStageTest(tf.test.TestCase, parameterized.TestCase):
  """Abstract base class for tests of `EncodingStageInterface`.

  Tests for each implementation `EncodingStageInterface` class should implement
  this class, and add additional tests specific to the behavior of the tested
  implementation.

  This class contains basic tests, which every implementation of
  `EncodingStageInterface` is expected to pass. It also contains set of testing
  utilities.

  In particular, the `test_one_to_many_encode_decode` and
  `test_many_to_one_encode_decode` methods ensure the implementation does not
  assume something that is not possible in scenarios where the class is meant to
  be used.
  """

  # -----------------
  # Abstract methods
  # -----------------
  @abc.abstractproperty
  def is_lossless(self):
    """Returns True if the encoding stage is lossless.

    That is, if the `EncodingStageInterface` returned by
    `default_encoding_stage` is such that encoding and decoding amounts to an
    identity.

    This property is used to determine whether to perform additional checks in
    the test methods.
    """
    pass

  @abc.abstractmethod
  def default_encoding_stage(self):
    """Provides a default constructor for an encoding stage.

    This is used for tests in the base class, which every implementation of
    `EncodingStageInterface` is expected to pass.

    Returns:
      An instance of a concrete `EncodingStageInterface` to be tested.
    """
    pass

  @abc.abstractmethod
  def default_input(self):
    """Provides a default input for testing the encoding.

    This is used for tests in the base class, which every implementation of
    EncodingStageInterface is expected to pass.

    The `shape` of the returned `Tensor` must be statically known.

    Returns:
      A `Tensor` object to be used as default testing input for encoding.
    """
    pass

  @abc.abstractmethod
  def common_asserts_for_test_data(self, data):
    """A collection of assertions for the results of encoding and decoding.

    This method takes a `TestData` object and evaluates any user provided
    expectations on the values. This method is used in multiple test methods and
    should not use TensorFlow in any way, only perform the assertions.

    Args:
      data: A `TestData` tuple containing numpy values with results to be
        evaluated.
    """
    pass

  # -------------
  # Test methods
  # -------------
  def test_default_encoding_stage(self):
    """Tests that `default_encoding_stage` return `EncodingStageInterface`."""
    stage = self.default_encoding_stage()
    self.assertIsInstance(stage, encoding_stage.EncodingStageInterface)

  def test_encoding_stage_constructor_does_not_modify_graph(self):
    """Tests that the constructor of encoding stage does not modify graph."""
    graph_def = tf.get_default_graph().as_graph_def()
    self.default_encoding_stage()
    new_graph_def = tf.get_default_graph().as_graph_def()
    tf.test.assert_equal_graph_def(graph_def, new_graph_def)

  def test_default_input_is_tensor_with_fully_defined_shape(self):
    """Tests that `default_input` returns a `Tesnor` of fully defined shape."""
    x = self.default_input()
    self.assertIsInstance(x, tf.Tensor)
    self.assertTrue(x.shape.is_fully_defined())

  def test_basic_encode_decode(self):
    """Tests the core functionality.

    This test method uses the default encoding stage and default input, executes
    encoding and decoding in the context of the same graph, and finally performs
    custom asserts on the resulting data.
    """
    # Get Tensors representing the encoded and decoded values and perform
    # generic type assertions.
    x = self.default_input()
    stage = self.default_encoding_stage()
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = TestData(x, encoded_x, decoded_x)
    self.generic_asserts(test_data, stage)

    # Evaluate the Tensors and get numpy values.
    test_data = self.evaluate(test_data)
    if self.is_lossless:
      self.assertAllClose(test_data.x, test_data.decoded_x)
    self.common_asserts_for_test_data(test_data)

  def test_one_to_many_encode_decode(self):
    """Tests the core functionality in the 'one-to-many' case.

    This method tests that the implementation can be used in a setting, where
    the encoding happens in one location, decoding happens in anohter location,
    and communication between these happens outside of TensorFlow.

    In particular, this ensures that the implementation does not create
    something incompatible with the use case, such as creating a TensorFlow
    state during encoding, and accessing it during decoding.
    """
    # This just delegates to a utility, which can be used if the same needs to
    # be tested with an input Tensor of specific properties, such as statically
    # unknown shape, potentially with addional assertions.
    test_data = self.run_one_to_many_encode_decode(
        self.default_encoding_stage(), self.default_input)
    self.common_asserts_for_test_data(test_data)

  def test_many_to_one_encode_decode(self):
    """Tests the core functionality in the 'many-to-one' case.

    This method tests that the implementation can be used in a setting, where
    the parameters are created in on location, communicated to a number of other
    locations, where different inputs are encoded, and decoding happens in the
    original location. The communication between these happens outside of
    TensorFlow.

    In particular, this ensures that the implementation does not create
    something incompatible with the use case, such as creating a TensorFlow
    state during encoding, and accessing it during decoding.
    """
    stage = self.default_encoding_stage()
    input_values = self.evaluate([self.default_input() for _ in range(3)])
    server_test_data, decode_params = self.run_many_to_one_encode_decode(
        stage, input_values)

    if self.is_lossless:
      self.assertAllClose(
          np.sum([d.x for d in server_test_data], axis=0),
          np.sum([d.decoded_x for d in server_test_data], axis=0))
    if stage.commutes_with_sum:
      self.assert_commutes_with_sum(server_test_data, stage, decode_params,
                                    input_values[0].shape)
    self.asserts_for_test_many_to_one_encode_decode(server_test_data)

  # ------------------
  # Testing utilities
  # ------------------
  def encode_decode_x(self, stage, x, encode_params, decode_params):
    """Given params, encodes and decodes input `Tensor`.

    Args:
      stage: An `EncodingStageInterface` to be used for encoding and decoding.
      x: A `Tensor` to be encoded and decoded.
      encode_params: Parameters to be provided to `stage.encode`
      decode_params: Parameters to be provided to `stage.decode`

    Returns:
      A tuple (encoded_x, decoded_x), where these are:
        encoded_x: A dictionary of `Tensor` objects representing the encoded
          input `x`.
        decoded_x: A single `Tensor`, representing decoded `encoded_x`.
    """
    encoded_x = stage.encode(x, encode_params)
    shape = None
    if stage.decode_needs_input_shape:
      shape = utils.static_or_dynamic_shape(x)
    decoded_x = stage.decode(encoded_x, decode_params, shape)
    return encoded_x, decoded_x

  def run_one_to_many_encode_decode(self, stage, input_fn):
    """Runs encoding and decoding in the one-to-many setting.

    This method creates the input `Tensor` in the context of one graph, creates
    and evaluates the encoded structure, along with `decode_params`. These are
    used as Python constants in another graph to create and evaluate decoding.

    Args:
      stage: An `EncodingStageInterface` to be used for encoding.
      input_fn: A callable object without arguments that creates and returns an
        input `Tensor` to be used for encoding.

    Returns:
      A `TestData` tuple containing numpy values representing the results.
    """
    server_graph = tf.Graph()
    with server_graph.as_default():
      x = input_fn()
      encode_params, decode_params = stage.get_params()
      encoded_x = stage.encode(x, encode_params)
      shape = utils.static_or_dynamic_shape(x)

    # Get all values out of TensorFlow as Python constants. This is a trivial
    # example of communication happening outside of TensorFlow.
    with self.session(graph=server_graph):
      x, decode_params, encoded_x, shape = self.evaluate_tf_py_list(
          [x, decode_params, encoded_x, shape])

    client_graph = tf.Graph()
    with client_graph.as_default():
      decoded_x = stage.decode(encoded_x, decode_params, shape=shape)

    with self.session(graph=client_graph):
      decoded_x = self.evaluate(decoded_x)

    return TestData(x, encoded_x, decoded_x)

  def run_many_to_one_encode_decode(self, stage, input_values):
    """Runs encoding and decoding in the many-to-one setting.

    This method creates and evaluates the parameters in the context of one
    graph, which are used to create and evaluate encoding in a new graph for
    every input value provided. These values are then decoded in the context of
    the first graph. If the provided `stage` commutes with sum, this is in
    addition verified.

    Args:
      stage: An `EncodingStageInterface` to be used for encoding.
      input_values: A list of numpy values to be used for encoding. All must
        have the same shape.

    Returns:
      A tuple `(server_test_data, decode_params)` where these are:
      server_test_data: A `list` of `TestData` tuples containing numpy values
        representing the results of encoding for each element of `input_values`.
      decode_params: Numpy values of the decode parameters used. These are
        values that should be used if additional decoding is to be done, such as
        for `assert_commutes_with_sum`.
    """
    server_graph = tf.Graph()
    with server_graph.as_default():
      shape = input_values[0].shape
      encode_params, decode_params = stage.get_params()

    with self.session(server_graph) as sess:
      encode_params, decode_params = self.evaluate_tf_py_list(
          [encode_params, decode_params], sess)

    client_test_data = []
    for x in input_values:
      client_graph = tf.Graph()
      with client_graph.as_default():
        encoded_x = stage.encode(x, encode_params)

      with self.session(client_graph):
        encoded_x = self.evaluate(encoded_x)
        client_test_data.append(TestData(x, encoded_x, None))

    server_test_data = []
    with server_graph.as_default():
      with self.session(server_graph) as sess:
        for test_data in client_test_data:
          decoded_x = stage.decode(
              test_data.encoded_x, decode_params, shape=shape)
          server_test_data.append(
              TestData(test_data.x, test_data.encoded_x, sess.run(decoded_x)))

    return server_test_data, decode_params

  def evaluate_tf_py_list(self, fetches, session=None):
    """Evaluates only provided `Tensor` objects and returns numpy values.

    Different from `self.evaluate` or `session.run`, which only takes TensorFlow
    objects to be evaluated, this method can take a combination of Python and
    TensorFlow objects, separates them, evaluates only the TensorFlow objects,
    and merges the resulting numpy values back with the original python values.

    Args:
      fetches: A `list` of fetches to be evalutated.
      session: An optional `tf.Session` object to be used for evaluation, if
        necessary to explicitly specify. If `None`, the default session will be
        used.

    Returns:
      A list of the same structure as `fetches`, with TensorFlow objects
      replaced by the result of single call to `self.evaluate` (or
      `session.run`) with these TensorFlow objects as the input.
    """
    # Split the fetches to two structures.
    py_fetches, tf_fetches = [], []
    placeholder_empty_tuple = ()
    assert isinstance(fetches, list), 'fetches should be a list.'
    for fetch in fetches:
      if isinstance(fetch, dict):
        d_py, d_tf = utils.split_dict_py_tf(fetch)
        py_fetches.append(d_py)
        tf_fetches.append(d_tf)
      elif tensor_util.is_tensor(fetch):
        py_fetches.append(None)
        tf_fetches.append(fetch)
      else:
        py_fetches.append(fetch)
        # This empty tuple is here as a marker to retain the value from
        # py_fetches, while keeping the list length same for simplicity of
        # reconstruction. This is effectively None, but self.evaluate does not
        # accept None as an input argument.
        tf_fetches.append(placeholder_empty_tuple)

    # Evaluate the structure containing the TensorFlow objects.
    if any((tensor_util.is_tensor(t) for t in nest.flatten(tf_fetches))):
      # Only evaluate something if there is a Tensor to be evaluated.
      if session:
        eval_fetches = session.run(tf_fetches)
      else:
        eval_fetches = self.evaluate(tf_fetches)
    else:
      eval_fetches = tf_fetches

    # Merge back the two structures, not contatining Tensors.
    for i, value in enumerate(eval_fetches):
      if isinstance(value, dict):
        eval_fetches[i] = utils.merge_dicts(value, py_fetches[i])
      elif value == placeholder_empty_tuple:
        eval_fetches[i] = py_fetches[i]
    return eval_fetches

  def generic_asserts(self, test_data, stage):
    """Collection of static checks every implementation is expected to satisfy.

    Args:
      test_data: A `TestData` tuple. All values should contain `Tensor` objects.
      stage: An `EncodingStageInterface` that generated the `test_data`.
    """
    # Every key in compressible_tensors_keys should be in encoded_x.
    for key in stage.compressible_tensors_keys:
      self.assertIn(key, test_data.encoded_x)

    # The return structure of encode should only contain Tensor objects, and no
    # Python constants.
    for tensor in six.itervalues(test_data.encoded_x):
      self.assertIsInstance(tensor, tf.Tensor)

    # With a statically known input shape, the shape of decoded_x should be
    # statically known. If not statically known, both should be unknown.
    self.assertEqual(test_data.x.shape, test_data.decoded_x.shape)

  def asserts_for_test_many_to_one_encode_decode(self, data):
    """Additional asserts for `test_many_to_one_encode_decode` method.

    By default, this method simply calls `common_asserts_for_test_data` on every
    element of `data`, but can be overridden by an implemented to provide custom
    or additional checks.

    Args:
      data: A `list` of `TestData` tuples containing numpy values to be used for
        the assertions.
    """
    for d in data:
      self.common_asserts_for_test_data(d)

  def assert_commutes_with_sum(self,
                               server_test_data,
                               stage,
                               decode_params,
                               shape=None):
    """Asserts that provided `EncodingStageInterface` commutes with sum.

    Given a list of `TestData` namedtuples containing numpy values of input and
    corresponding encoded and decoded values, makes sure that the sum of the
    decoded values is the same as first summing encoded values, and then
    decoding.

    Args:
      server_test_data: A `list` of `TestData` namedtuples.
      stage: An `EncodingStageInterface` object that was used to generate
        `server_test_data` and is to be used in the assert.
      decode_params: Parameters to be used for decoding by `stage`. Must be the
        same values as used for generating `server_test_data`.
      shape: An optional shape for the `decode` method of `stage`.
    """
    # This assert should be only used with an instance that commutes with sum.
    assert stage.commutes_with_sum

    expected_sum = np.sum([d.decoded_x for d in server_test_data], axis=0)
    sum_encoded_x = {}
    for k in server_test_data[0].encoded_x:
      sum_encoded_x[k] = np.sum(
          [d.encoded_x[k] for d in server_test_data], axis=0)
    with tf.Graph().as_default():
      with self.session() as sess:
        decode_sum_encoded_x = sess.run(
            stage.decode(sum_encoded_x, decode_params, shape))
    self.assertAllClose(expected_sum, decode_sum_encoded_x)


class PlusOneEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, adding 1.

  This is the simplest example implementation of an `EncodingStageInterface` -
  no state, no constructor arguments, no shape information needed for decoding,
  no commutativity with sum.
  """

  def __init__(self):
    self._plus_value = 1.0

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return ['values']

  @property
  def commutes_with_sum(self):
    """See base class."""
    return False

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  def get_params(self, name=None):
    """See base class."""
    params = {'add': tf.constant(self._plus_value)}
    return params, params

  @encoding_stage.tf_style_encode('plus_one_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    return {'values': x + encode_params['add']}

  @encoding_stage.tf_style_decode('plus_one_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    return encoded_tensors['values'] - decode_params['add']


class TimesTwoEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, multiplying by 2.

  This is an example implementation of an `EncodingStageInterface` that commutes
  with sum.
  """

  def __init__(self):
    self._factor = 2.0

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return ['values']

  @property
  def commutes_with_sum(self):
    """See base class."""
    return True

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  def get_params(self, name=None):
    """See base class."""
    params = {'factor': tf.constant(self._factor)}
    return params, params

  @encoding_stage.tf_style_encode('times_two_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    return {'values': x * encode_params['factor']}

  @encoding_stage.tf_style_decode('times_two_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    return encoded_tensors['values'] / decode_params['factor']


class SimpleLinearEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, computing a simple linear transformation.

  This is an example implementation of an `EncodingStageInterface` that can take
  constructor arguments, which can be both python constants, or `tf.Variable`
  objects, and subsequently expose those via `encode_params` / `decode_params`.
  """

  def __init__(self, a, b):
    self._a = a
    self._b = b

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return ['values']

  @property
  def commutes_with_sum(self):
    """See base class."""
    return False

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  def get_params(self, name=None):
    """See base class."""
    params = {'a': self._a, 'b': self._b}
    return params, params

  @encoding_stage.tf_style_encode('simple_linear_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    return {'values': encode_params['a'] * x + encode_params['b']}

  @encoding_stage.tf_style_decode('simple_linear_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    return (
        (encoded_tensors['values'] - decode_params['b']) / decode_params['a'])


class ReduceMeanEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, computing a mean and remembering original shape.

  This is an example implementation of an `EncodingStageInterface` that requires
  the original shape information for decoding.
  """

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return ['values']

  @property
  def commutes_with_sum(self):
    """See base class."""
    return True

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return True

  def get_params(self, name=None):
    """See base class."""
    return {}, {}

  @encoding_stage.tf_style_encode('reduce_mean_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    del encode_params  # Unused.
    encoded_tensors = {'values': tf.reduce_mean(x, keepdims=True)}
    if not x.shape.is_fully_defined():
      encoded_tensors['shape'] = tf.shape(x)
    return encoded_tensors

  @encoding_stage.tf_style_decode('reduce_mean_decode')
  def decode(self, encoded_tensors, decode_params, shape, name=None):
    """See base class."""
    del decode_params  # Unused.
    return tf.tile(encoded_tensors['values'], shape)


class RandomAddSubtractOneEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, randomly adding or subtracting 1.

  This is an example implementation of an `EncodingStageInterface` that is
  lossless, but unbiased on expectation. This is a propery of a variety
  implementations of the interface, and this class serves as an example of how
  the unbiasedness can be tested.
  """

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return ['values']

  @property
  def commutes_with_sum(self):
    """See base class."""
    return True

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  def get_params(self, name=None):
    """See base class."""
    return {}, {}

  @encoding_stage.tf_style_encode('plus_one_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    del encode_params
    return {'values': x + tf.sign(tf.random.normal(tf.shape(x)))}

  @encoding_stage.tf_style_decode('plus_one_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    del decode_params
    return encoded_tensors['values']


# TODO(b/120977011): Implement a deterministic random op that can be checked in
# third_party and provide an example implementation of EncodingStageInterface
# that relies on a shared random seed.


def get_tensor_with_random_shape(expected_num_elements=10,
                                 source_fn=tf.random.uniform):
  """Returns a 1-D `Tensor` with random shape.

  The `Tensor` is created by creating a `Tensor` with `2*expected_num_elements`
  and inlcude each element in the rerurned `Tensor` with probability `0.5`.
  Thus, the returned `Tensor` has unknown, and non-deterministic shape.

  Args:
    expected_num_elements: The number of elements the returned `Tensor` should
      have on expectation.
    source_fn: A Python callable that generates values for the returned
      `Tensor`.

  Returns:
    A 1-D `Tensor` with random shape.
  """
  return tf.squeeze(
      tf.gather(
          source_fn([2 * expected_num_elements]),
          tf.where(
              tf.less(tf.random_uniform([2 * expected_num_elements]), 0.5))))
