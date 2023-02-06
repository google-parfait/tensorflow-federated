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

from typing import Any, Optional

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import chunkers

# Convenience Aliases
_CharacterEncoding = chunkers.CharacterEncoding


class CreateChunkerTest(tf.test.TestCase):

  def test_utf8_encoding(self):
    chunker = chunkers.create_chunker(
        string_max_bytes=1234, encoding=_CharacterEncoding.UTF8
    )

    self.assertIsInstance(chunker, chunkers.UTF8Chunker)
    self.assertEqual(chunker.dtype, tf.int64)

  def test_utf8_int32(self):
    chunker = chunkers.create_chunker(
        string_max_bytes=1234, encoding=_CharacterEncoding.UTF8, dtype=tf.int32
    )

    self.assertEqual(chunker.dtype, tf.int32)

  def test_unknown_encoding(self):
    chunker = chunkers.create_chunker(
        string_max_bytes=1234, encoding=_CharacterEncoding.UNKNOWN
    )

    self.assertIsInstance(chunker, chunkers.BinaryChunker)

  def test_unknown_encoding_wrong_dtype_raises(self):
    with self.assertRaisesRegex(ValueError, 'dtype'):
      chunkers.create_chunker(
          string_max_bytes=1234,
          encoding=_CharacterEncoding.UNKNOWN,
          dtype=tf.uint16,
      )


# https://en.wikipedia.org/wiki/UTF-8#Codepage_layout
INVALID_UNICODE_CODEPOINT = bytes.fromhex('FF')


class BinaryChunkerValidationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('none', None, TypeError),
      ('wrong_type', object(), TypeError),
      ('zero', 0, ValueError),
      ('negative', -1, ValueError),
      ('too_large', tf.int32.max + 1, ValueError),
  )
  def test_init_invalid_string_max_bytes_raises(
      self, string_max_bytes: Any, expected_exception: BaseException
  ):
    with self.assertRaisesRegex(expected_exception, 'string_max_bytes'):
      chunkers.BinaryChunker(string_max_bytes=string_max_bytes, dtype=tf.int32)

  @parameterized.named_parameters(
      ('wrong_type', object(), TypeError),
      ('negative', -1, ValueError),
      ('too_small', 100, ValueError),
      ('too_large', tf.int32.max + 1, ValueError),
  )
  def test_init_invalid_max_chunk_value_raises(
      self, max_chunk_value: Any, expected_exception: BaseException
  ):
    with self.assertRaisesRegex(expected_exception, 'max_chunk_value'):
      chunkers.BinaryChunker(
          string_max_bytes=10, max_chunk_value=max_chunk_value, dtype=tf.int32
      )

  @parameterized.named_parameters(
      ('none', None, TypeError),
      ('wrong_type', object(), TypeError),
      ('unsupported', tf.int16, ValueError),
  )
  def test_init_invalid_dtype_raises(
      self, dtype: Any, expected_exception: BaseException
  ):
    with self.assertRaisesRegex(expected_exception, 'dtype'):
      chunkers.BinaryChunker(string_max_bytes=1234, dtype=dtype)

  @parameterized.named_parameters(
      ('none', None),
      ('not_tensor', object()),
      ('wrong_dtype', [1, 2, 3]),
      ('wrong_shape', [['foo', 'bar']]),
  )
  def test_encode_tensorflow_invalid_input_raises(self, input_strings: Any):
    chunker = chunkers.BinaryChunker(string_max_bytes=1234, dtype=tf.int32)

    with self.assertRaises((TypeError, ValueError)):
      chunker.encode_tensorflow(input_strings)

  @parameterized.named_parameters(
      ('none', None),
      ('not_tensor', object()),
      ('wrong_rank', np.array([[[1, 7]]], dtype=np.int32)),
      ('wrong_shape', np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)),
  )
  def test_decode_tensorflow_invalid_input_raises(self, encoded_inputs: Any):
    chunker = chunkers.BinaryChunker(string_max_bytes=1, dtype=tf.int32)

    with self.assertRaisesRegex(ValueError, 'encoded_chunks'):
      chunker.decode_tensorflow(encoded_inputs)


_BINARY_TEST_INPUTS = [
    b'lmnop',
    b'QQQQQqqqqq',
    b' a b c ',
    '😍😻'.encode(),
    'अ'.encode(),
    b'',
    INVALID_UNICODE_CODEPOINT * 8,
]


@parameterized.named_parameters(
    dict(
        testcase_name='int32_1byte_chunk_size',
        dtype=tf.int32,
        max_chunk_value=255,
    ),
    dict(
        testcase_name='int32_small_chunk_size',
        dtype=tf.int32,
        max_chunk_value=1000,
    ),
    dict(
        testcase_name='int32_max_chunk_size',
        dtype=tf.int32,
        max_chunk_value=None,
    ),
    dict(
        testcase_name='int64_small_chunk_size',
        dtype=tf.int64,
        max_chunk_value=2000,
    ),
    dict(
        testcase_name='int64_int32_chunk_size',
        dtype=tf.int64,
        max_chunk_value=tf.int32.max,
    ),
    dict(
        testcase_name='int64_max_chunk_size',
        dtype=tf.int64,
        max_chunk_value=None,
    ),
)
class BinaryChunkerTest(tf.test.TestCase, parameterized.TestCase):

  def test_dtype(self, dtype: tf.dtypes.DType, max_chunk_value: Optional[int]):
    chunker = chunkers.BinaryChunker(
        string_max_bytes=20, dtype=dtype, max_chunk_value=max_chunk_value
    )

    self.assertEqual(chunker.dtype, dtype)

  def test_encode_and_decode_tensorflow_as_expected(
      self, dtype: tf.dtypes.DType, max_chunk_value: Optional[int]
  ):
    input_binary = _BINARY_TEST_INPUTS
    string_max_bytes = 10

    chunker = chunkers.BinaryChunker(
        string_max_bytes=string_max_bytes,
        dtype=dtype,
        max_chunk_value=max_chunk_value,
    )
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_binary)
    )
    decoded_strings = chunker.decode_tensorflow(encoded_chunks).numpy()

    self.assertCountEqual(input_binary, decoded_strings)
    self.assertCountEqual(trimmed_input_strings.numpy(), decoded_strings)

  def test_encode_and_decode_tensorflow_trim_strings_as_expected(
      self, dtype: tf.dtypes.DType, max_chunk_value: Optional[int]
  ):
    input_binary = _BINARY_TEST_INPUTS
    string_max_bytes = 5
    expected_trimmed = [x[:string_max_bytes] for x in input_binary]

    chunker = chunkers.BinaryChunker(
        string_max_bytes=string_max_bytes,
        dtype=dtype,
        max_chunk_value=max_chunk_value,
    )
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_binary)
    )
    decoded_strings = chunker.decode_tensorflow(encoded_chunks).numpy()

    self.assertCountEqual(expected_trimmed, decoded_strings)
    self.assertCountEqual(trimmed_input_strings.numpy(), decoded_strings)

  def test_encode_and_decode_tensorflow_long_strings(
      self, dtype: tf.dtypes.DType, max_chunk_value: Optional[int]
  ):
    string_max_bytes = max_chunk_value or dtype.max
    # `string_max_bytes = max_chunk_value` causes OOMs for very large values
    string_max_bytes = min(string_max_bytes, 1_000_000)
    input_binary = [b'x' * string_max_bytes]
    expected_trimmed = [x[:string_max_bytes] for x in input_binary]

    chunker = chunkers.BinaryChunker(
        string_max_bytes=string_max_bytes,
        dtype=dtype,
        max_chunk_value=max_chunk_value,
    )
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_binary)
    )
    decoded_strings = chunker.decode_tensorflow(encoded_chunks).numpy()

    self.assertCountEqual(expected_trimmed, decoded_strings)
    self.assertCountEqual(trimmed_input_strings.numpy(), decoded_strings)

  def test_decode_tensorflow_single_string(
      self, dtype: tf.dtypes.DType, max_chunk_value: Optional[int]
  ):
    # IbltDecoder calls `decode_tensorflow` with a single string at a time in
    # 1-D tensor, rather than 2-D tensor with all strings like
    # `encode_tensorflow` returns. Most tests feed the results of
    # `encode_tensorflow` directly into `decode_tensorflow`, but this test
    # verifies `decode_tensorflow` is compatible with `IbltDecoder`.

    input_binary = [b' a b c ']
    string_max_bytes = 10
    chunker = chunkers.BinaryChunker(
        string_max_bytes=string_max_bytes,
        dtype=dtype,
        max_chunk_value=max_chunk_value,
    )

    encoded_chunks, _ = chunker.encode_tensorflow(tf.constant(input_binary))
    single_encoded_string = tf.squeeze(encoded_chunks[:1])
    assert single_encoded_string.shape == [chunker.get_num_chunks()]

    decoded_binary = chunker.decode_tensorflow(single_encoded_string).numpy()

    self.assertCountEqual(input_binary, decoded_binary)


class UTF8ChunkerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('int32', tf.int32),
      ('int64', tf.int64),
  )
  def test_dtype(self, dtype: tf.dtypes.DType):
    chunker = chunkers.UTF8Chunker(string_max_bytes=1234, dtype=dtype)

    self.assertEqual(chunker.dtype, dtype)

  @parameterized.named_parameters(
      ('int32_10', 10, tf.int32),
      ('int64_10', 10, tf.int64),
      ('int32_20', 20, tf.int32),
      ('int64_20', 20, tf.int64),
  )
  def test_encode_and_decode_tensorflow_as_expected(
      self, string_max_bytes, dtype
  ):
    input_strings = [
        '',
        'some',
        'unicodes',
        'अ',
        'क',
        'æ',
        '☺️',
        '☺️',
        '☺️',
        '😇',
        ' has space ',
        'has, comma',
    ]

    chunker = chunkers.UTF8Chunker(string_max_bytes, dtype=dtype)
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings)
    )
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
      (
          'max_len_8',
          8,
          [
              '',
              'some',
              'interesting',
              'unicodes',
              ' has more space ',
              'has, some, comma',
              '新年快乐',
              '☺️😇',
          ],
          [
              '',
              'some',
              'interest',
              'unicodes',
              ' has mor',
              'has, som',
              '新年',
              '☺️',
          ],
      ),
      (
          'max_len_5',
          5,
          [
              '',
              'some',
              'interesting',
              'unicodes',
              ' has more space ',
              'has, some, comma',
              '新年快乐',
              '☺️😇',
          ],
          [
              '',
              'some',
              'interest',
              'unicodes',
              ' has mor',
              'has, som',
              '新年',
              '☺️',
          ],
      ),
      (
          'max_len_10',
          10,
          [
              '',
              'some',
              'interesting',
              'unicodes',
              ' has more space ',
              'has, some, comma',
              '新年快乐',
              '☺️😇',
          ],
          [
              '',
              'some',
              'interesting',
              'unicodes',
              ' has more space ',
              'has, some, comma',
              '新年快乐',
              '☺️😇',
          ],
      ),
  )
  def test_encode_and_decode_tensorflow_trim_strings_as_expected(
      self, string_max_bytes, input_strings, expected_decoded_strings
  ):
    dtype = tf.int64

    chunker = chunkers.UTF8Chunker(string_max_bytes, dtype=dtype)
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings)
    )
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
  def test_encode_and_decode_python_as_expected(self, string_max_bytes, dtype):
    input_strings = [
        '',
        'some',
        'unicodes',
        'अ',
        'क',
        'æ',
        '☺️',
        '☺️',
        '☺️',
        '😇',
        'has space ',
        'has, comma',
    ]

    chunker = chunkers.UTF8Chunker(string_max_bytes, dtype=dtype)
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings)
    )
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
      (
          'max_len_8',
          8,
          [
              '',
              'some',
              'interesting',
              'unicodes',
              ' has more space ',
              'has, some, comma',
              '新年快乐',
              '☺️😇',
          ],
          [
              '',
              'some',
              'interest',
              'unicodes',
              ' has mor',
              'has, som',
              '新年',
              '☺️',
          ],
      ),
      (
          'max_len_5',
          5,
          [
              '',
              'some',
              'interesting',
              'unicodes',
              ' has more space ',
              'has, some, comma',
              '新年快乐',
              '☺️😇',
          ],
          [
              '',
              'some',
              'interest',
              'unicodes',
              ' has mor',
              'has, som',
              '新年',
              '☺️',
          ],
      ),
      (
          'max_len_10',
          10,
          [
              '',
              'some',
              'interesting',
              'unicodes',
              ' has more space ',
              'has, some, comma',
              '新年快乐',
              '☺️😇',
          ],
          [
              '',
              'some',
              'interesting',
              'unicodes',
              ' has more space ',
              'has, some, comma',
              '新年快乐',
              '☺️😇',
          ],
      ),
  )
  def test_encode_and_decode_python_trim_strings_as_expected(
      self, string_max_bytes, input_strings, expected_decoded_strings
  ):
    dtype = tf.int64

    chunker = chunkers.UTF8Chunker(string_max_bytes, dtype=dtype)
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings)
    )
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
      self, string_max_bytes, dtype
  ):
    max_chunk_value = 2**31 - 1
    input_strings = [
        '',
        'some',
        'unicodes',
        'अ',
        'क',
        'æ',
        '☺️',
        '☺️',
        '☺️',
        '😇',
        ' has space ',
        'has, comma',
    ]

    chunker = chunkers.UTF8Chunker(
        string_max_bytes, max_chunk_value=max_chunk_value, dtype=dtype
    )
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings)
    )
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
      self, string_max_bytes, dtype
  ):
    max_chunk_value = 2**31 - 1
    input_strings = [
        '',
        'some',
        'unicodes',
        'अ',
        'क',
        'æ',
        '☺️',
        '☺️',
        '☺️',
        '😇',
        ' has space ',
        'has, comma',
    ]

    chunker = chunkers.UTF8Chunker(
        string_max_bytes, max_chunk_value=max_chunk_value, dtype=dtype
    )
    encoded_chunks, trimmed_input_strings = chunker.encode_tensorflow(
        tf.constant(input_strings)
    )
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

  def test_arguments_string_max_bytes_neg_value_error(self):
    with self.assertRaises(ValueError):
      chunkers.UTF8Chunker(string_max_bytes=-1, dtype=tf.int64)

  def test_arguments_max_chunk_neg_value_error(self):
    with self.assertRaises(ValueError):
      chunkers.UTF8Chunker(
          string_max_bytes=10, max_chunk_value=-1, dtype=tf.int64
      )

  def test_max_chunk_value_too_large_error(self):
    with self.assertRaises(ValueError):
      chunkers.UTF8Chunker(
          string_max_bytes=10, max_chunk_value=2**33, dtype=tf.int32
      )

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
      chunkers.UTF8Chunker(string_max_bytes=10, dtype=dtype)


if __name__ == '__main__':
  tf.test.main()
