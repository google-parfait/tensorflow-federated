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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import chunkers


class UTF8ChunkerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('int32_10', 10, tf.int32),
      ('int64_10', 10, tf.int64),
      ('int32_20', 20, tf.int32),
      ('int64_20', 20, tf.int64),
  )
  def test_encode_and_decode_tensorflow_as_expected(self, string_max_length,
                                                    dtype):
    input_strings = [
        '', 'some', 'unicodes', 'अ', 'क', 'æ', '☺️', '☺️',
        '☺️', '😇', ' has space ', 'has, comma'
    ]

    chunker = chunkers.UTF8Chunker(string_max_length, dtype=dtype)
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings))
    decoded_strings = chunker.decode_tensorflow(encoded_chunks).numpy()

    # Set 'ignore' in `.decode()` to ignore decoding error because the strings
    # are trimmed when they are encoded, and the trimming might cut in the
    # middle of a multi-byte utf-8 character.
    decoded_strings_list = [
        decoded_strings[i].decode('utf-8', 'ignore')
        for i in range(decoded_strings.shape[0])
    ]

    trimmed_input_strings = trimmed_input_strings.numpy()

    self.assertCountEqual(input_strings, decoded_strings_list)
    self.assertCountEqual(trimmed_input_strings, decoded_strings)

  @parameterized.named_parameters(
      ('max_len_8', 8, [
          '', 'some', 'interesting', 'unicodes', ' has more space ',
          'has, some, comma', '新年快乐', '☺️😇'
      ], [
          '', 'some', 'interest', 'unicodes', ' has mor', 'has, som', '新年',
          '☺️'
      ]),
      ('max_len_5', 5, [
          '', 'some', 'interesting', 'unicodes', ' has more space ',
          'has, some, comma', '新年快乐', '☺️😇'
      ], [
          '', 'some', 'interest', 'unicodes', ' has mor', 'has, som', '新年',
          '☺️'
      ]),
      ('max_len_10', 10, [
          '', 'some', 'interesting', 'unicodes', ' has more space ',
          'has, some, comma', '新年快乐', '☺️😇'
      ], [
          '', 'some', 'interesting', 'unicodes', ' has more space ',
          'has, some, comma', '新年快乐', '☺️😇'
      ]),
  )
  def test_encode_and_decode_tensorflow_trim_strings_as_expected(
      self, string_max_length, input_strings, expected_decoded_strings):
    dtype = tf.int64

    chunker = chunkers.UTF8Chunker(string_max_length, dtype=dtype)
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings))
    decoded_strings = chunker.decode_tensorflow(encoded_chunks).numpy()
    decoded_strings_list = [
        decoded_strings[i].decode('utf-8', 'ignore')
        for i in range(decoded_strings.shape[0])
    ]

    trimmed_input_strings = trimmed_input_strings.numpy()

    self.assertCountEqual(expected_decoded_strings, decoded_strings_list)
    self.assertCountEqual(trimmed_input_strings, decoded_strings)

  @parameterized.named_parameters(
      ('int32_10', 10, tf.int32),
      ('int64_10', 10, tf.int64),
      ('int32_20', 20, tf.int32),
      ('int64_20', 20, tf.int64),
  )
  def test_encode_and_decode_python_as_expected(self, string_max_length, dtype):
    input_strings = [
        '', 'some', 'unicodes', 'अ', 'क', 'æ', '☺️', '☺️',
        '☺️', '😇', 'has space ', 'has, comma'
    ]

    chunker = chunkers.UTF8Chunker(string_max_length, dtype=dtype)
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings))
    decoded_strings = chunker.decode_python(encoded_chunks.numpy())

    # Set 'ignore' in `.decode()` to ignore decoding error because the strings
    # are trimmed when they are encoded, and the trimming might cut in the
    # middle of a multi-byte utf-8 character.
    decoded_strings_list = [
        decoded_strings[i].decode('utf-8', 'ignore')
        for i in range(decoded_strings.shape[0])
    ]

    trimmed_input_strings = trimmed_input_strings.numpy()

    self.assertCountEqual(input_strings, decoded_strings_list)
    self.assertCountEqual(trimmed_input_strings, decoded_strings)

  @parameterized.named_parameters(
      ('max_len_8', 8, [
          '', 'some', 'interesting', 'unicodes', ' has more space ',
          'has, some, comma', '新年快乐', '☺️😇'
      ], [
          '', 'some', 'interest', 'unicodes', ' has mor', 'has, som', '新年',
          '☺️'
      ]),
      ('max_len_5', 5, [
          '', 'some', 'interesting', 'unicodes', ' has more space ',
          'has, some, comma', '新年快乐', '☺️😇'
      ], [
          '', 'some', 'interest', 'unicodes', ' has mor', 'has, som', '新年',
          '☺️'
      ]),
      ('max_len_10', 10, [
          '', 'some', 'interesting', 'unicodes', ' has more space ',
          'has, some, comma', '新年快乐', '☺️😇'
      ], [
          '', 'some', 'interesting', 'unicodes', ' has more space ',
          'has, some, comma', '新年快乐', '☺️😇'
      ]),
  )
  def test_encode_and_decode_python_trim_strings_as_expected(
      self, string_max_length, input_strings, expected_decoded_strings):
    dtype = tf.int64

    chunker = chunkers.UTF8Chunker(string_max_length, dtype=dtype)
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings))
    decoded_strings = chunker.decode_python(encoded_chunks.numpy())

    # Set 'ignore' in `.decode()` to ignore decoding error because the strings
    # are trimmed when they are encoded, and the trimming might cut in the
    # middle of a multi-byte utf-8 character.
    decoded_strings_list = [
        decoded_strings[i].decode('utf-8', 'ignore')
        for i in range(decoded_strings.shape[0])
    ]

    trimmed_input_strings = trimmed_input_strings.numpy()

    self.assertCountEqual(expected_decoded_strings, decoded_strings_list)
    self.assertCountEqual(trimmed_input_strings, decoded_strings)

  @parameterized.named_parameters(
      ('int32_10', 10, tf.int32),
      ('int64_10', 10, tf.int64),
      ('int32_20', 20, tf.int32),
      ('int64_20', 20, tf.int64),
  )
  def test_encode_and_decode_tensorflow_small_bytes_per_chunk_as_expected(
      self, string_max_length, dtype):
    max_chunk_value = 2**31 - 1
    input_strings = [
        '', 'some', 'unicodes', 'अ', 'क', 'æ', '☺️', '☺️',
        '☺️', '😇', ' has space ', 'has, comma'
    ]

    chunker = chunkers.UTF8Chunker(
        string_max_length, max_chunk_value=max_chunk_value, dtype=dtype)
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings))
    self.assertAllLess(encoded_chunks, max_chunk_value)

    decoded_strings = chunker.decode_tensorflow(encoded_chunks).numpy()

    # Set 'ignore' in `.decode()` to ignore decoding error because the strings
    # are trimmed when they are encoded, and the trimming might cut in the
    # middle of a multi-byte utf-8 character.
    decoded_strings_list = [
        decoded_strings[i].decode('utf-8', 'ignore')
        for i in range(decoded_strings.shape[0])
    ]

    trimmed_input_strings = trimmed_input_strings.numpy()

    self.assertCountEqual(input_strings, decoded_strings_list)
    self.assertCountEqual(trimmed_input_strings, decoded_strings)

  @parameterized.named_parameters(
      ('int32_10', 10, tf.int32),
      ('int64_10', 10, tf.int64),
      ('int32_20', 20, tf.int32),
      ('int64_20', 20, tf.int64),
  )
  def test_encode_and_decode_python_small_bytes_per_chunk_as_expected(
      self, string_max_length, dtype):
    max_chunk_value = 2**31 - 1
    input_strings = [
        '', 'some', 'unicodes', 'अ', 'क', 'æ', '☺️', '☺️',
        '☺️', '😇', ' has space ', 'has, comma'
    ]

    chunker = chunkers.UTF8Chunker(
        string_max_length, max_chunk_value=max_chunk_value, dtype=dtype)
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings))
    self.assertAllLess(encoded_chunks, max_chunk_value)

    decoded_strings = chunker.decode_python(encoded_chunks.numpy())

    # Set 'ignore' in `.decode()` to ignore decoding error because the strings
    # are trimmed when they are encoded, and the trimming might cut in the
    # middle of a multi-byte utf-8 character.
    decoded_strings_list = [
        decoded_strings[i].decode('utf-8', 'ignore')
        for i in range(decoded_strings.shape[0])
    ]

    trimmed_input_strings = trimmed_input_strings.numpy()

    self.assertCountEqual(input_strings, decoded_strings_list)
    self.assertCountEqual(trimmed_input_strings, decoded_strings)

  def test_arguments_string_max_length_neg_value_error(self):
    with self.assertRaises(ValueError):
      chunkers.UTF8Chunker(string_max_length=-1, dtype=tf.int64)

  def test_arguments_max_chunk_neg_value_error(self):
    with self.assertRaises(ValueError):
      chunkers.UTF8Chunker(
          string_max_length=10, max_chunk_value=-1, dtype=tf.int64)

  def test_max_chunk_value_too_large_error(self):
    with self.assertRaises(ValueError):
      chunkers.UTF8Chunker(
          string_max_length=10, max_chunk_value=2**33, dtype=tf.int32)

  @parameterized.named_parameters(
      ('uint16', tf.uint16),
      ('float32', tf.float32),
      ('bool', tf.bool),
      ('string', tf.string),
      ('qint8', tf.qint8),
      ('variant', tf.variant),
  )
  def test_arguments_dtype_value_error(self, dtype):
    with self.assertRaises(ValueError):
      chunkers.UTF8Chunker(string_max_length=10, dtype=dtype)


if __name__ == '__main__':
  tf.test.main()
