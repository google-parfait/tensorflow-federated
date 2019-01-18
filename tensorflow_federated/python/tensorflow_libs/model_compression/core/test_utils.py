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

from absl.testing import parameterized
import numpy as np
import six
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs.model_compression.core import encoding_stage
from tensorflow_federated.python.tensorflow_libs.model_compression.core import utils

DEFAULT_RTOL = 1e-05
DEFAULT_ATOL = 1e-05

# Named tuple containing the values summarizing the results for a single
# evaluation of an EncodingStageInterface or an AdaptiveEncodingStageInterface.
TestData = collections.namedtuple(
    'TestData',
    [
        'x',  # The input provided to encoding.
        'encoded_x',  # A dictionary of values representing the encoded input x.
        'decoded_x',  # Decoded value. Has the same shape as x.
        # The fields below are only relevant for AdaptiveEncodingStageInterface,
        # and will not be populated while testing an EncodingStageInterface.
        'initial_state',  # Initial state used for encoding.
        'state_update_tensors',  # State update tensors created by encoding.
        'updated_state',  # Updated state after encoding.
    ])
# Set the dafault values to be None, to enable use of TestData while testing
# EncodingStageInterface, without needing to be aware of the other fields.
TestData.__new__.__defaults__ = (None,) * len(TestData._fields)


# This metaclass enables adding abc.ABCMeta metaclass to a class inheriting from
# parameterized.TestCase.
class ParameterizedABCMeta(abc.ABCMeta, parameterized.TestGeneratorMetaclass):
  pass


@six.add_metaclass(ParameterizedABCMeta)
class BaseEncodingStageTest(tf.test.TestCase, parameterized.TestCase):
  """Abstract base class for testing encoding stage implementations.

  Tests for each implementation of `EncodingStageInterface` and
  `AdaptiveEncodingStageInterface` should implement this class, and add
  additional tests specific to the behavior of the tested implementation.

  This class contains basic tests, which every implementation of
  `EncodingStageInterface` is expected to pass, and it contains a set of
  utilities for testing.

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
    """Tests the correctness of `default_encoding_stage`."""
    stage = self.default_encoding_stage()
    self.assertIsInstance(stage,
                          (encoding_stage.EncodingStageInterface,
                           encoding_stage.AdaptiveEncodingStageInterface))

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
    if is_adaptive_stage(stage):
      state = stage.initial_state()
      encode_params, decode_params = stage.get_params(state)
      encoded_x, decoded_x, state_update_tensors = self.encode_decode_x(
          stage, x, encode_params, decode_params)
      updated_state = stage.update_state(state, state_update_tensors)
      test_data = TestData(x, encoded_x, decoded_x, state, state_update_tensors,
                           updated_state)
    else:
      encode_params, decode_params = stage.get_params()
      encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                  decode_params)
      test_data = TestData(x, encoded_x, decoded_x)
    self.generic_asserts(test_data, stage)

    # Evaluate the Tensors and get numpy values.
    test_data = self.evaluate_test_data(test_data)
    if self.is_lossless:
      self.assertAllClose(
          test_data.x,
          test_data.decoded_x,
          rtol=DEFAULT_RTOL,
          atol=DEFAULT_ATOL)
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
          np.sum([d.decoded_x for d in server_test_data], axis=0),
          rtol=DEFAULT_RTOL,
          atol=DEFAULT_ATOL)
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
      stage: An `EncodingStageInterface` or an `AdaptiveEncodingStageInterface`
        to be used for encoding and decoding.
      x: A `Tensor` to be encoded and decoded.
      encode_params: Parameters to be provided to `stage.encode`
      decode_params: Parameters to be provided to `stage.decode`

    Returns:
      A tuple (encoded_x, decoded_x) if `stage` is an `EncodingStageInterface`,
      or a tuple (encoded_x, decoded_x, state_update_tensors) if `stage` is an
      `AdaptiveEncodingStageInterface`, where these are:
        encoded_x: A dictionary of `Tensor` objects representing the encoded
          input `x`.
        decoded_x: A single `Tensor`, representing decoded `encoded_x`.
        state_update_tensors: A dictionary of `Tensor` objects representing the
          information necessary for updating the state.
    """
    if is_adaptive_stage(stage):
      encoded_x, state_update_tensors = stage.encode(x, encode_params)
    else:
      encoded_x = stage.encode(x, encode_params)

    shape = None
    if stage.decode_needs_input_shape:
      shape = utils.static_or_dynamic_shape(x)
    decoded_x = stage.decode(encoded_x, decode_params, shape)

    if is_adaptive_stage(stage):
      return encoded_x, decoded_x, state_update_tensors
    else:
      return encoded_x, decoded_x

  def run_one_to_many_encode_decode(self, stage, input_fn, state=None):
    """Runs encoding and decoding in the one-to-many setting.

    This method creates the input `Tensor` in the context of one graph, creates
    and evaluates the encoded structure, along with `decode_params`. These are
    used as Python constants in another graph to create and evaluate decoding.

    The need for `input_fn`, as opposed to a simple numpy constant, is because
    some stages need to work with `Tensor` objects that do not have statically
    known shape. Such `Tensor` needs to be created in the context of the graph
    in which it is to be evaluated, that is, inside of this method.

    Args:
      stage: An `EncodingStageInterface` or an `AdaptiveEncodingStageInterface`
        to be used for encoding.
      input_fn: A callable object without arguments that creates and returns a
        `Tensor` or numpy value to be used for encoding.
      state: A dictionary representing the state. Can be set only if `stage` is
        an `AdaptiveEncodingStageInterface`.

    Returns:
      A `TestData` tuple containing numpy values representing the results.
    """

    def _adaptive_one_to_many_encode_decode(state):
      """Implementation of the method for `AdaptiveEncodingStageInterface`."""
      server_graph = tf.Graph()
      with server_graph.as_default():
        x = input_fn()
        shape = utils.static_or_dynamic_shape(x)
        if state is None:
          state = stage.initial_state()
        encode_params, decode_params = stage.get_params(state)
        encoded_x, state_update_tensors = stage.encode(x, encode_params)
        updated_state = stage.update_state(state, state_update_tensors)

      # Get all values out of TensorFlow as Python constants. This is a trivial
      # example of communication happening outside of TensorFlow.
      with self.session(graph=server_graph):
        (x, decode_params, encoded_x, state, state_update_tensors,
         updated_state, shape) = self.evaluate_tf_py_list([
             x, decode_params, encoded_x, state, state_update_tensors,
             updated_state, shape
         ])

      client_graph = tf.Graph()
      with client_graph.as_default():
        decoded_x = stage.decode(encoded_x, decode_params, shape=shape)
      with self.session(graph=client_graph):
        decoded_x = self.evaluate(decoded_x)

      return TestData(x, encoded_x, decoded_x, state, state_update_tensors,
                      updated_state)

    def _non_adaptive_one_to_many_encode_decode():
      """Implementation of the method for `EncodingStageInterface`."""
      server_graph = tf.Graph()
      with server_graph.as_default():
        x = input_fn()
        shape = utils.static_or_dynamic_shape(x)
        encode_params, decode_params = stage.get_params()
        encoded_x = stage.encode(x, encode_params)

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

    if is_adaptive_stage(stage):
      return _adaptive_one_to_many_encode_decode(state)
    else:
      assert state is None
      return _non_adaptive_one_to_many_encode_decode()

  def run_many_to_one_encode_decode(self, stage, input_values, state=None):
    """Runs encoding and decoding in the many-to-one setting.

    This method creates and evaluates the parameters in the context of one
    graph, which are used to create and evaluate encoding in a new graph for
    every input value provided. These values are then decoded in the context of
    the first graph. If the provided `stage` commutes with sum, this is in
    addition verified.

    Args:
      stage: An `EncodingStageInterface` or an `AdaptiveEncodingStageInterface`
        to be used for encoding.
      input_values: A list of numpy values to be used for encoding. All must
        have the same shape.
      state: A dictionary representing the state. Can be set only if `stage` is
        an `AdaptiveEncodingStageInterface`.

    Returns:
      A tuple `(server_test_data, decode_params)` where these are:
      server_test_data: A `list` of `TestData` tuples containing numpy values
        representing the results of encoding for each element of `input_values`.
      decode_params: Numpy values of the decode parameters used. These are
        values that should be used if additional decoding is to be done, such as
        for `assert_commutes_with_sum`.
    """

    def _adaptive_many_to_one_encode_decode(state):
      """Implementation of the method for `AdaptiveEncodingStageInterface`."""
      server_graph = tf.Graph()
      with server_graph.as_default():
        shape = input_values[0].shape
        if state is None:
          state = stage.initial_state()
        encode_params, decode_params = stage.get_params(state)
      with self.session(server_graph) as sess:
        encode_params, decode_params, state = self.evaluate_tf_py_list(
            [encode_params, decode_params, state], sess)

      client_test_data = []
      for x in input_values:
        client_graph = tf.Graph()
        with client_graph.as_default():
          encoded_x, state_update_tensors = stage.encode(x, encode_params)
        with self.session(client_graph):
          encoded_x, state_update_tensors = self.evaluate(
              [encoded_x, state_update_tensors])
          client_test_data.append(
              TestData(x, encoded_x, state_update_tensors=state_update_tensors))

      server_test_data = []
      with server_graph.as_default():
        with self.session(server_graph) as sess:
          for test_data in client_test_data:
            decoded_x = stage.decode(
                test_data.encoded_x, decode_params, shape=shape)
            server_test_data.append(
                test_data._replace(
                    decoded_x=sess.run(decoded_x), initial_state=state))
          # Compute and append the updated state to all TestData objects.
          all_state_update_tensors = [
              d.state_update_tensors for d in server_test_data
          ]
          aggregated_state_update_tensors = aggregate_state_update_tensors(
              stage, all_state_update_tensors)
          updated_state = sess.run(
              stage.update_state(state, aggregated_state_update_tensors))
          server_test_data = [
              d._replace(updated_state=updated_state) for d in server_test_data
          ]

      return server_test_data, decode_params

    def _non_adaptive_many_to_one_encode_decode():
      """Implementation of the method for `EncodingStageInterface`."""
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
          client_test_data.append(TestData(x, encoded_x))

      server_test_data = []
      with server_graph.as_default():
        with self.session(server_graph) as sess:
          for test_data in client_test_data:
            decoded_x = stage.decode(
                test_data.encoded_x, decode_params, shape=shape)
            server_test_data.append(
                test_data._replace(decoded_x=sess.run(decoded_x)))
      return server_test_data, decode_params

    if is_adaptive_stage(stage):
      return _adaptive_many_to_one_encode_decode(state)
    else:
      assert state is None
      return _non_adaptive_many_to_one_encode_decode()

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
      elif tf.contrib.framework.is_tensor(fetch):
        py_fetches.append(None)
        tf_fetches.append(fetch)
      else:
        py_fetches.append(fetch)
        # This empty tuple is here as a marker to retain the value from
        # py_fetches, while keeping the list length same for simplicity of
        # reconstruction. This is effectively None, but self.evaluate does not
        # accept None as an input argument.
        tf_fetches.append(placeholder_empty_tuple)

    eval_fetches = self.maybe_evaluate(tf_fetches, session)
    # Merge back the two structures, not contatining Tensors.
    for i, value in enumerate(eval_fetches):
      if isinstance(value, dict):
        eval_fetches[i] = utils.merge_dicts(value, py_fetches[i])
      elif value == placeholder_empty_tuple:
        eval_fetches[i] = py_fetches[i]
    return eval_fetches

  def evaluate_test_data(self, test_data, session=None):
    """Evaluates a `TestData` object.

    Args:
      test_data: A `TestData` namedtuple.
      session: Optional. A `tf.Session` object in the context of which the
        evaluation is to happen.

    Returns:
      A new `TestData` object with `Tensor` objects in `test_data` replaced by
      numpy values.

    Raises:
      TypeError: If `test_data` is not a `TestData` namedtuple.
    """
    if not isinstance(test_data, TestData):
      raise TypeError('A TestData object must be provided.')
    _, data_tf = utils.split_dict_py_tf(test_data._asdict())
    return test_data._replace(**self.maybe_evaluate(data_tf, session))

  def maybe_evaluate(self, fetches, session=None):
    """Evaluates `fetches`, if containing any `Tensor` objects.

    Args:
      fetches: Any nested structure compatible with `tf.contrib.framework.nest`.
      session: Optional. A `tf.Session` object in the context of which the
        evaluation is to happen.

    Returns:
      `fetches` with any `Tensor` objects replaced by numpy values.
    """
    if any((tf.contrib.framework.is_tensor(t)
            for t in tf.contrib.framework.nest.flatten(fetches))):
      if session:
        fetches = session.run(fetches)
      else:
        fetches = self.evaluate(fetches)
    return fetches

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

    if is_adaptive_stage(stage):
      # The property should have keys matching those of state_update_tensors.
      self.assertSameElements(stage.state_update_aggregation_modes.keys(),
                              test_data.state_update_tensors.keys())

      for tensor in six.itervalues(test_data.initial_state):
        self.assertTrue(tf.contrib.framework.is_tensor(tensor))
      for tensor in six.itervalues(test_data.state_update_tensors):
        self.assertTrue(tf.contrib.framework.is_tensor(tensor))
      for tensor in six.itervalues(test_data.updated_state):
        self.assertTrue(tf.contrib.framework.is_tensor(tensor))

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
      sum_encoded_x[k] = np.sum([d.encoded_x[k] for d in server_test_data],
                                axis=0)
    with tf.Graph().as_default():
      with self.session() as sess:
        decode_sum_encoded_x = sess.run(
            stage.decode(sum_encoded_x, decode_params, shape))
    self.assertAllClose(
        expected_sum,
        decode_sum_encoded_x,
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL)


class PlusOneEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, adding 1.

  This is the simplest example implementation of an `EncodingStageInterface` -
  no state, no constructor arguments, no shape information needed for decoding,
  no commutativity with sum.
  """

  ENCODED_VALUES_KEY = 'values'
  ADD_PARAM_KEY = 'add'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

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
    with tf.name_scope(name, 'plus_one_get_params'):
      params = {self.ADD_PARAM_KEY: tf.constant(1.0)}
      return params, params

  @encoding_stage.tf_style_encode('plus_one_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    return {self.ENCODED_VALUES_KEY: x + encode_params[self.ADD_PARAM_KEY]}

  @encoding_stage.tf_style_decode('plus_one_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    decoded_x = (
        encoded_tensors[self.ENCODED_VALUES_KEY] -
        decode_params[self.ADD_PARAM_KEY])
    return decoded_x


class TimesTwoEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, multiplying by 2.

  This is an example implementation of an `EncodingStageInterface` that commutes
  with sum.
  """

  ENCODED_VALUES_KEY = 'values'
  FACTOR_PARAM_KEY = 'factor'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

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
    with tf.name_scope(name, 'times_two_get_params'):
      params = {self.FACTOR_PARAM_KEY: tf.constant(2.0)}
      return params, params

  @encoding_stage.tf_style_encode('times_two_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    return {self.ENCODED_VALUES_KEY: x * encode_params[self.FACTOR_PARAM_KEY]}

  @encoding_stage.tf_style_decode('times_two_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    decoded_x = (
        encoded_tensors[self.ENCODED_VALUES_KEY] /
        decode_params[self.FACTOR_PARAM_KEY])
    return decoded_x


class SimpleLinearEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, computing a simple linear transformation.

  This is an example implementation of an `EncodingStageInterface` that can take
  constructor arguments, which can be both python constants, or `tf.Variable`
  objects, and subsequently expose those via `encode_params` / `decode_params`.
  """

  ENCODED_VALUES_KEY = 'values'
  A_PARAM_KEY = 'a_param'
  B_PARAM_KEY = 'b_param'

  def __init__(self, a, b):
    self._a = a
    self._b = b

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

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
    params = {self.A_PARAM_KEY: self._a, self.B_PARAM_KEY: self._b}
    return params, params

  @encoding_stage.tf_style_encode('simple_linear_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    a, b = encode_params[self.A_PARAM_KEY], encode_params[self.B_PARAM_KEY]
    return {self.ENCODED_VALUES_KEY: a * x + b}

  @encoding_stage.tf_style_decode('simple_linear_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    a, b = decode_params[self.A_PARAM_KEY], decode_params[self.B_PARAM_KEY]
    return (encoded_tensors[self.ENCODED_VALUES_KEY] - b) / a


class ReduceMeanEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, computing a mean and remembering original shape.

  This is an example implementation of an `EncodingStageInterface` that requires
  the original shape information for decoding.

  Note that the encoding does not store the shape in the return structure of the
  `encode` method. Instead, the shape information will be handled separately by
  the higher level `Encoder`.
  """

  ENCODED_VALUES_KEY = 'values'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

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
    return {self.ENCODED_VALUES_KEY: tf.reduce_mean(x, keepdims=True)}

  @encoding_stage.tf_style_decode('reduce_mean_decode')
  def decode(self, encoded_tensors, decode_params, shape, name=None):
    """See base class."""
    del decode_params  # Unused.
    return tf.tile(encoded_tensors[self.ENCODED_VALUES_KEY], shape)


class RandomAddSubtractOneEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, randomly adding or subtracting 1.

  This is an example implementation of an `EncodingStageInterface` that is not
  lossless, but unbiased on expectation. This is a propery of a variety
  implementations of the interface, and this class serves as an example of how
  the unbiasedness can be tested.
  """

  ENCODED_VALUES_KEY = 'values'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

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

  @encoding_stage.tf_style_encode('add_subtract_one_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    del encode_params  # Unused.
    return {self.ENCODED_VALUES_KEY: x + tf.sign(tf.random.normal(tf.shape(x)))}

  @encoding_stage.tf_style_decode('add_subtract_one_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    del decode_params  # Unused.
    return encoded_tensors[self.ENCODED_VALUES_KEY]


class SignIntFloatEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, encoding input into multiple outputs.

  This is an example implementation of an `EncodingStageInterface` that is
  losless and splits the input into three components - the integer part, the
  floating part and the signs.
  """

  ENCODED_SIGNS_KEY = 'signs'
  ENCODED_INTS_KEY = 'ints'
  ENCODED_FLOATS_KEY = 'floats'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [
        self.ENCODED_SIGNS_KEY, self.ENCODED_INTS_KEY, self.ENCODED_FLOATS_KEY
    ]

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
    return {}, {}

  @encoding_stage.tf_style_encode('sign_int_float_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    del encode_params  # Unused.
    signs = tf.sign(x)
    abs_vals = tf.abs(x)
    ints = tf.floor(abs_vals)
    floats = abs_vals - ints
    return {
        self.ENCODED_SIGNS_KEY: signs,
        self.ENCODED_INTS_KEY: ints,
        self.ENCODED_FLOATS_KEY: floats
    }

  @encoding_stage.tf_style_decode('sign_int_float_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    del decode_params  # Unused.
    signs = encoded_tensors[self.ENCODED_SIGNS_KEY]
    ints = encoded_tensors[self.ENCODED_INTS_KEY]
    floats = encoded_tensors[self.ENCODED_FLOATS_KEY]
    return signs * (ints + floats)


def dummy_rng_source(seed, num_elements):
  """Dummy TensorFlow random number generator.

  We need a custom random source, which would be always deterministic given a
  random seed. That is not currently available available in TensorFlow. This
  simple function serves an illustrative purpose. It is *not* a useful random
  number generator, and should only be used in tests.

  Args:
    seed: A random seed.
    num_elements: Number of random values to generate.

  Returns:
    A `Tensor` of shape `(num_elements)` containing pseudorandom values.
  """

  def next_num(num):
    # This creates a cycle of length 136.
    return tf.mod((num * 13), 137)

  num = tf.reshape(tf.mod(seed, 136) + 1, (1,))
  result = num
  for _ in range(num_elements - 1):
    num = next_num(num)
    result = tf.concat([result, num], 0)
  return tf.to_float(result)


class PlusRandomNumEncodingStage(encoding_stage.EncodingStageInterface):
  """[Example] encoding stage, adding random values given a random seed.

  This is an example implementation of an `EncodingStageInterface` that depends
  on a shared random seed. The seed `Tensor` should be created in the
  `get_params` method, and the same values should evantually be passed to both
  `encode` and `decode` methods, making sure a randomized transform is
  invertible.
  """

  ENCODED_VALUES_KEY = 'values'
  SEED_PARAM_KEY = 'seed'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

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
    with tf.name_scope(name, 'plus_random_num_get_params'):
      params = {
          self.SEED_PARAM_KEY:
              tf.random.uniform((), maxval=tf.int32.max, dtype=tf.int32)
      }
      return params, params

  @encoding_stage.tf_style_encode('plus_random_num_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    addend = dummy_rng_source(encode_params[self.SEED_PARAM_KEY],
                              x.shape.num_elements())
    addend = tf.reshape(addend, x.shape)
    return {self.ENCODED_VALUES_KEY: x + addend}

  @encoding_stage.tf_style_decode('plus_random_num_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    x = encoded_tensors[self.ENCODED_VALUES_KEY]
    addend = dummy_rng_source(decode_params[self.SEED_PARAM_KEY],
                              x.shape.num_elements())
    addend = tf.reshape(addend, x.shape)
    return x - addend


class PlusNSquaredEncodingStage(encoding_stage.AdaptiveEncodingStageInterface):
  """[Example] adaptive encoding stage, adding N*N in N-th iteration.

  This is an example implementation of an `AdaptiveEncodingStageInterface` that
  modifies state, which controls the creation of params. This is also a simple
  example of how an `EncodingStageInterface` can be wrapped as an
  `AdaptiveEncodingStageInterface`, without modifying the wrapped encode and
  decode methods.
  """

  ENCODED_VALUES_KEY = PlusOneEncodingStage.ENCODED_VALUES_KEY
  ADD_PARAM_KEY = PlusOneEncodingStage.ADD_PARAM_KEY
  ITERATION_STATE_KEY = 'iteration'

  def __init__(self):
    self._stage = PlusOneEncodingStage()

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

  @property
  def commutes_with_sum(self):
    """See base class."""
    return False

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  @property
  def state_update_aggregation_modes(self):
    """See base class."""
    return {}

  def initial_state(self, name=None):
    """See base class."""
    with tf.name_scope(name, 'plus_n_squared_initial_state'):
      return {self.ITERATION_STATE_KEY: tf.constant(1.0)}

  def update_state(self, state, state_update_tensors, name=None):
    """See base class."""
    del state_update_tensors  # Unused.
    with tf.name_scope(name, 'plus_n_squared_update_state'):
      return {
          self.ITERATION_STATE_KEY:
              state[self.ITERATION_STATE_KEY] + tf.constant(1.0)
      }

  def get_params(self, state, name=None):
    """See base class."""
    with tf.name_scope(name, 'plus_n_squared_get_params'):
      params = {
          self.ADD_PARAM_KEY:
              state[self.ITERATION_STATE_KEY] * state[self.ITERATION_STATE_KEY]
      }
      return params, params

  @encoding_stage.tf_style_encode('plus_n_squared_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    return self._stage.encode(x, encode_params, name), {}

  @encoding_stage.tf_style_decode('plus_n_squared_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    return self._stage.decode(encoded_tensors, decode_params, shape, name)


class AdaptiveNormalizeEncodingStage(
    encoding_stage.AdaptiveEncodingStageInterface):
  """[Example] encoding stage, adaptively normalizing data.

  This is an example implementation of an `AdaptiveEncodingStageInterface` that
  updates the state based on information stored in `state_update_tensors`. This
  implementation wraps `TimesTwoEncodingStage`, and adaptively changes the
  parameters that control the `encode` and `decode` methods.

  It assumes that over iterations, the input values to be encoded come from
  certain static distribution, and tries to find a good factor to normalize the
  input to be of unit norm.
  """

  ENCODED_VALUES_KEY = TimesTwoEncodingStage.ENCODED_VALUES_KEY
  FACTOR_PARAM_KEY = TimesTwoEncodingStage.FACTOR_PARAM_KEY
  FACTOR_STATE_KEY = 'factor'
  NORM_STATE_UPDATE_KEY = 'norm'

  def __init__(self):
    self._stage = TimesTwoEncodingStage()

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

  @property
  def commutes_with_sum(self):
    """See base class."""
    return True

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  @property
  def state_update_aggregation_modes(self):
    """See base class."""
    return {
        self.NORM_STATE_UPDATE_KEY: encoding_stage.StateAggregationMode.STACK
    }

  def initial_state(self, name=None):
    """See base class."""
    with tf.name_scope(name, 'adaptive_normalize_initial_state'):
      return {self.FACTOR_STATE_KEY: tf.constant(1.0)}

  # pylint: disable=g-doc-args,g-doc-return-or-yield
  def update_state(self, state, state_update_tensors, name=None):
    """Updates the state (see base class).

    This method illustrates how the implementation can handle state update based
    on a single encoding, or based on a multiple encodings collectively.

    As specified by `self.state_update_aggregation_modes`, the
    `NORM_STATE_UPDATE_KEY` from `state_update_tensors` are to be stacked. That
    means, that the corresponding input to this method should be a `Tensor` with
    each element corresponding to a single output of an encoding. So this can be
    a single element, in the one-to-many setting, or multiple elements, in the
    many-to-one setting.

    The `update_state` method thus can compute arbitrary function of the
    relevant values. In this case, it maintains a rolling average of previous
    states, where the weight to be used depends on the number of updates
    received. Note that the specific implementation is not necessarily useful or
    efficient; it rather serves as an illustration of what can be done.
    """
    with tf.name_scope(name, 'adaptive_normalize_update_state'):
      # This can be either a Tensor or a numpy value. Number of elements can be
      # computed for both as product of the elements of the shape vector.
      num_updates = np.prod(
          state_update_tensors[self.NORM_STATE_UPDATE_KEY].shape)
      norm_mean = tf.reduce_mean(
          state_update_tensors[self.NORM_STATE_UPDATE_KEY])
      weight = 0.9**num_updates  # Use a stronger weight for more updates.
      new_factor = (
          weight * state[self.FACTOR_STATE_KEY] + (1 - weight) / norm_mean)
      return {self.FACTOR_STATE_KEY: new_factor}

  def get_params(self, state, name=None):
    """See base class."""
    with tf.name_scope(name, 'adaptive_normalize_get_params'):
      params = {self.FACTOR_PARAM_KEY: state[self.FACTOR_STATE_KEY]}
      return params, params

  @encoding_stage.tf_style_encode('times_n_encode')
  def encode(self, x, encode_params, name=None):
    """See base class."""
    return (self._stage.encode(x, encode_params, name), {
        self.NORM_STATE_UPDATE_KEY: tf.norm(x)
    })

  @encoding_stage.tf_style_decode('times_n_decode')
  def decode(self, encoded_tensors, decode_params, shape=None, name=None):
    """See base class."""
    return self._stage.decode(encoded_tensors, decode_params, shape, name)


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


def is_adaptive_stage(stage):
  """Returns `True` if `stage` is an `AdaptiveEncodingStageInterface`."""
  if isinstance(stage, encoding_stage.EncodingStageInterface):
    assert not isinstance(stage, encoding_stage.AdaptiveEncodingStageInterface)
    return False
  elif isinstance(stage, encoding_stage.AdaptiveEncodingStageInterface):
    return True
  else:
    raise TypeError(
        'The provided `stage` must be either `EncodingStageInterface` or '
        '`AdaptiveEncodingStageInterface`.')


def aggregate_state_update_tensors(stage, state_update_tensors):
  """Aggregates a collection of values for state update.

  This method in an trivial example of implementation of the aggregation modes,
  when all the values are available as numpy values simultaneously.

  Args:
    stage: An `AdaptiveEncodingStageInterface` object.
    state_update_tensors: A `list` of `dict` objects, each of which corresponds
      to `state_update_tensors` generated by the `stage.encode` method. Each
      dictionary thus needs to have the same structure, corresponding to
      `stage.state_update_aggregation_modes`, and contain numpy values.

  Returns:
    A dictionary of aggregated values.

  Raises:
    TypeError: If `stage` is not an `AdaptiveEncodingStageInterface`.
  """

  def _aggregate(values, aggregation_mode):
    """Aggregates values according to aggregation mode."""
    if aggregation_mode == encoding_stage.StateAggregationMode.SUM:
      return np.sum(np.stack(values), axis=0)
    elif aggregation_mode == encoding_stage.StateAggregationMode.MAX:
      return np.amax(np.stack(values), axis=0)
    elif aggregation_mode == encoding_stage.StateAggregationMode.MIN:
      return np.amin(np.stack(values), axis=0)
    elif aggregation_mode == encoding_stage.StateAggregationMode.STACK:
      return np.stack(values)

  if not is_adaptive_stage(stage):
    raise TypeError(
        'The provided `stage` must be an `AdaptiveEncodingStageInterface`.')
  aggregated_state_update_tensors = {}
  for key, mode in six.iteritems(stage.state_update_aggregation_modes):
    aggregated_state_update_tensors[key] = _aggregate(
        [t[key] for t in state_update_tensors], mode)
  return aggregated_state_update_tensors
