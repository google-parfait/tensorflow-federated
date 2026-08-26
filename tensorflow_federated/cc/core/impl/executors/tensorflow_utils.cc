/* Copyright 2023, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"

#include <complex>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "Eigen/Core"
#include "google/protobuf/repeated_ptr_field.h"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"

namespace tensorflow_federated {

absl::StatusOr<federated_language::DataType> DataTypeFromTensorFlowDataType(
    tensorflow::DataType data_type_pb) {
  switch (data_type_pb) {
    case tensorflow::DataType::DT_BOOL:
      return federated_language::DataType::DT_BOOL;
    case tensorflow::DataType::DT_INT8:
      return federated_language::DataType::DT_INT8;
    case tensorflow::DataType::DT_INT16:
      return federated_language::DataType::DT_INT16;
    case tensorflow::DataType::DT_INT32:
      return federated_language::DataType::DT_INT32;
    case tensorflow::DataType::DT_INT64:
      return federated_language::DataType::DT_INT64;
    case tensorflow::DataType::DT_UINT8:
      return federated_language::DataType::DT_UINT8;
    case tensorflow::DataType::DT_UINT16:
      return federated_language::DataType::DT_UINT16;
    case tensorflow::DataType::DT_UINT32:
      return federated_language::DataType::DT_UINT32;
    case tensorflow::DataType::DT_UINT64:
      return federated_language::DataType::DT_UINT64;
    case tensorflow::DataType::DT_HALF:
      return federated_language::DataType::DT_HALF;
    case tensorflow::DataType::DT_FLOAT:
      return federated_language::DataType::DT_FLOAT;
    case tensorflow::DataType::DT_DOUBLE:
      return federated_language::DataType::DT_DOUBLE;
    case tensorflow::DataType::DT_COMPLEX64:
      return federated_language::DataType::DT_COMPLEX64;
    case tensorflow::DataType::DT_COMPLEX128:
      return federated_language::DataType::DT_COMPLEX128;
    case tensorflow::DataType::DT_BFLOAT16:
      return federated_language::DataType::DT_BFLOAT16;
    case tensorflow::DataType::DT_STRING:
      return federated_language::DataType::DT_STRING;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", data_type_pb));
  }
}

absl::StatusOr<tensorflow::DataType> TensorFlowDataTypeFromDataType(
    federated_language::DataType data_type_pb) {
  switch (data_type_pb) {
    case federated_language::DataType::DT_BOOL:
      return tensorflow::DataType::DT_BOOL;
    case federated_language::DataType::DT_INT8:
      return tensorflow::DataType::DT_INT8;
    case federated_language::DataType::DT_INT16:
      return tensorflow::DataType::DT_INT16;
    case federated_language::DataType::DT_INT32:
      return tensorflow::DataType::DT_INT32;
    case federated_language::DataType::DT_INT64:
      return tensorflow::DataType::DT_INT64;
    case federated_language::DataType::DT_UINT8:
      return tensorflow::DataType::DT_UINT8;
    case federated_language::DataType::DT_UINT16:
      return tensorflow::DataType::DT_UINT16;
    case federated_language::DataType::DT_UINT32:
      return tensorflow::DataType::DT_UINT32;
    case federated_language::DataType::DT_UINT64:
      return tensorflow::DataType::DT_UINT64;
    case federated_language::DataType::DT_HALF:
      return tensorflow::DataType::DT_HALF;
    case federated_language::DataType::DT_FLOAT:
      return tensorflow::DataType::DT_FLOAT;
    case federated_language::DataType::DT_DOUBLE:
      return tensorflow::DataType::DT_DOUBLE;
    case federated_language::DataType::DT_COMPLEX64:
      return tensorflow::DataType::DT_COMPLEX64;
    case federated_language::DataType::DT_COMPLEX128:
      return tensorflow::DataType::DT_COMPLEX128;
    case federated_language::DataType::DT_BFLOAT16:
      return tensorflow::DataType::DT_BFLOAT16;
    case federated_language::DataType::DT_STRING:
      return tensorflow::DataType::DT_STRING;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", data_type_pb));
  }
}

absl::StatusOr<federated_language::ArrayShape> ArrayShapeFromTensorShape(
    const tensorflow::TensorShape& tensor_shape) {
  federated_language::ArrayShape shape_pb;
  for (int i = 0; i < tensor_shape.dims(); i++) {
    shape_pb.mutable_dim()->Add(tensor_shape.dim_size(i));
  }
  shape_pb.set_unknown_rank(tensor_shape.unknown_rank());
  return shape_pb;
}

absl::StatusOr<tensorflow::TensorShape> TensorShapeFromArrayShape(
    const federated_language::ArrayShape& shape_pb) {
  if (shape_pb.unknown_rank()) {
    return absl::InvalidArgumentError(
        "Expected federated_language::ArrayShape to have a known rank, try "
        "constructing "
        "a tensorflow::PartialTensorShape using "
        "tensorflow_federated::PartialTensorShapeFromArrayShape instead.");
  }

  tensorflow::TensorShape shape;
  TFF_TRY(tensorflow::TensorShape::BuildTensorShape(shape_pb.dim(), &shape));
  return shape;
}

tensorflow::PartialTensorShape PartialTensorShapeFromArrayShape(
    const federated_language::ArrayShape& shape_pb) {
  if (!shape_pb.unknown_rank()) {
    return tensorflow::PartialTensorShape(shape_pb.dim());
  } else {
    return tensorflow::PartialTensorShape();
  }
}

absl::StatusOr<federated_language::Array> ArrayFromTensor(
    const tensorflow::Tensor& tensor) {
  federated_language::Array array_pb;
  federated_language::DataType data_type =
      TFF_TRY(DataTypeFromTensorFlowDataType(tensor.dtype()));
  array_pb.set_dtype(data_type);
  federated_language::ArrayShape shape_pb =
      TFF_TRY(ArrayShapeFromTensorShape(tensor.shape()));
  array_pb.mutable_shape()->Swap(&shape_pb);

  tensorflow::TensorProto tensor_pb;
  tensor.AsProtoField(&tensor_pb);

  switch (tensor_pb.dtype()) {
    case tensorflow::DataType::DT_BOOL: {
      array_pb.mutable_bool_list()->mutable_value()->Assign(
          tensor_pb.bool_val().begin(), tensor_pb.bool_val().end());
      break;
    }
    case tensorflow::DataType::DT_INT8: {
      array_pb.mutable_int8_list()->mutable_value()->Assign(
          tensor_pb.int_val().begin(), tensor_pb.int_val().end());
      break;
    }
    case tensorflow::DataType::DT_INT16: {
      array_pb.mutable_int16_list()->mutable_value()->Assign(
          tensor_pb.int_val().begin(), tensor_pb.int_val().end());
      break;
    }
    case tensorflow::DataType::DT_INT32: {
      array_pb.mutable_int32_list()->mutable_value()->Assign(
          tensor_pb.int_val().begin(), tensor_pb.int_val().end());
      break;
    }
    case tensorflow::DataType::DT_INT64: {
      array_pb.mutable_int64_list()->mutable_value()->Assign(
          tensor_pb.int64_val().begin(), tensor_pb.int64_val().end());
      break;
    }
    case tensorflow::DataType::DT_UINT8: {
      array_pb.mutable_uint8_list()->mutable_value()->Assign(
          tensor_pb.int_val().begin(), tensor_pb.int_val().end());
      break;
    }
    case tensorflow::DataType::DT_UINT16: {
      array_pb.mutable_uint16_list()->mutable_value()->Assign(
          tensor_pb.int_val().begin(), tensor_pb.int_val().end());
      break;
    }
    case tensorflow::DataType::DT_UINT32: {
      array_pb.mutable_uint32_list()->mutable_value()->Assign(
          tensor_pb.uint32_val().begin(), tensor_pb.uint32_val().end());
      break;
    }
    case tensorflow::DataType::DT_UINT64: {
      array_pb.mutable_uint64_list()->mutable_value()->Assign(
          tensor_pb.uint64_val().begin(), tensor_pb.uint64_val().end());
      break;
    }
    case tensorflow::DataType::DT_HALF: {
      array_pb.mutable_float16_list()->mutable_value()->Assign(
          tensor_pb.half_val().begin(), tensor_pb.half_val().end());
      break;
    }
    case tensorflow::DataType::DT_FLOAT: {
      array_pb.mutable_float32_list()->mutable_value()->Assign(
          tensor_pb.float_val().begin(), tensor_pb.float_val().end());
      break;
    }
    case tensorflow::DataType::DT_DOUBLE: {
      array_pb.mutable_float64_list()->mutable_value()->Assign(
          tensor_pb.double_val().begin(), tensor_pb.double_val().end());
      break;
    }
    case tensorflow::DataType::DT_COMPLEX64: {
      array_pb.mutable_complex64_list()->mutable_value()->Assign(
          tensor_pb.scomplex_val().begin(), tensor_pb.scomplex_val().end());
      break;
    }
    case tensorflow::DataType::DT_COMPLEX128: {
      array_pb.mutable_complex128_list()->mutable_value()->Assign(
          tensor_pb.dcomplex_val().begin(), tensor_pb.dcomplex_val().end());
      break;
    }
    case tensorflow::DataType::DT_BFLOAT16: {
      array_pb.mutable_bfloat16_list()->mutable_value()->Assign(
          tensor_pb.half_val().begin(), tensor_pb.half_val().end());
      break;
    }
    case tensorflow::DataType::DT_STRING: {
      array_pb.mutable_string_list()->mutable_value()->Assign(
          tensor_pb.string_val().begin(), tensor_pb.string_val().end());
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", tensor_pb.dtype()));
  }

  return array_pb;
}

template <typename T>
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<T>& src,
                                  T* dest) {
  absl::c_copy(src, dest);
}

// Overload for different SrcType and DestType.
template <typename SrcType, typename DestType>
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<SrcType>& src,
                                  DestType* dest) {
  absl::c_transform(src, dest, [](const SrcType& x) -> DestType {
    return static_cast<DestType>(x);
  });
}

// Overload for Eigen::half.
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<int32_t>& src,
                                  Eigen::half* dest) {
  // Values of dtype np.float16 are packed to and unpacked from a protobuf
  // field of type int32 using the following logic in order to maintain
  // compatibility with how other external environments (e.g. TensorFlow, Jax)
  // represent values of np.float16.
  absl::c_transform(src, dest, [](int32_t x) -> Eigen::half {
    return Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(x));
  });
}

// Overload for complex.
template <typename T>
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<T>& src,
                                  std::complex<T>* dest) {
  absl::c_copy(src, reinterpret_cast<T*>(dest));
}

// Overload for Eigen::bfloat16.
static void CopyFromRepeatedField(const google::protobuf::RepeatedField<int32_t>& src,
                                  Eigen::bfloat16* dest) {
  // Values of dtype ml_dtypes.bfloat16 are packed to and unpacked from a
  // protobuf field of type int32 using the following logic in order to maintain
  // compatibility with how other external environments (e.g. TensorFlow, Jax)
  // represent values of ml_dtypes.bfloat16.
  absl::c_transform(src, dest, [](int32_t x) -> Eigen::bfloat16 {
    return Eigen::numext::bit_cast<Eigen::bfloat16>(static_cast<uint16_t>(x));
  });
}

// Overload for string.
static void CopyFromRepeatedField(
    const google::protobuf::RepeatedPtrField<std::string>& src,
    tensorflow::tstring* dest) {
  absl::c_copy(src, dest);
}

absl::StatusOr<tensorflow::Tensor> TensorFromArray(
    const federated_language::Array& array_pb) {
  switch (array_pb.kind_case()) {
    case federated_language::Array::kBoolList: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<bool>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.bool_list().value(),
                            tensor.flat<bool>().data());
      return tensor;
    }
    case federated_language::Array::kInt8List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<int8_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.int8_list().value(),
                            tensor.flat<int8_t>().data());
      return tensor;
    }
    case federated_language::Array::kInt16List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<int16_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.int16_list().value(),
                            tensor.flat<int16_t>().data());
      return tensor;
    }
    case federated_language::Array::kInt32List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<int32_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.int32_list().value(),
                            tensor.flat<int32_t>().data());
      return tensor;
    }
    case federated_language::Array::kInt64List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<int64_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.int64_list().value(),
                            tensor.flat<int64_t>().data());
      return tensor;
    }
    case federated_language::Array::kUint8List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<uint8_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint8_list().value(),
                            tensor.flat<uint8_t>().data());
      return tensor;
    }
    case federated_language::Array::kUint16List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<uint16_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint16_list().value(),
                            tensor.flat<uint16_t>().data());
      return tensor;
    }
    case federated_language::Array::kUint32List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<uint32_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint32_list().value(),
                            tensor.flat<uint32_t>().data());
      return tensor;
    }
    case federated_language::Array::kUint64List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<uint64_t>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.uint64_list().value(),
                            tensor.flat<uint64_t>().data());
      return tensor;
    }
    case federated_language::Array::kFloat16List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<Eigen::half>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.float16_list().value(),
                            tensor.flat<Eigen::half>().data());
      return tensor;
    }
    case federated_language::Array::kFloat32List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<float>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.float32_list().value(),
                            tensor.flat<float>().data());
      return tensor;
    }
    case federated_language::Array::kFloat64List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<double>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.float64_list().value(),
                            tensor.flat<double>().data());
      return tensor;
    }
    case federated_language::Array::kComplex64List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<tensorflow::complex64>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.complex64_list().value(),
                            tensor.flat<tensorflow::complex64>().data());
      return tensor;
    }
    case federated_language::Array::kComplex128List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<tensorflow::complex128>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.complex128_list().value(),
                            tensor.flat<tensorflow::complex128>().data());
      return tensor;
    }
    case federated_language::Array::kBfloat16List: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<Eigen::bfloat16>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.bfloat16_list().value(),
                            tensor.flat<Eigen::bfloat16>().data());
      return tensor;
    }
    case federated_language::Array::kStringList: {
      tensorflow::Tensor tensor(
          tensorflow::DataTypeToEnum<tensorflow::tstring>::value,
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape())));
      CopyFromRepeatedField(array_pb.string_list().value(),
                            tensor.flat<tensorflow::tstring>().data());
      return tensor;
    }
    default: {
      tensorflow::TensorShape shape =
          TFF_TRY(TensorShapeFromArrayShape(array_pb.shape()));
      if (shape.num_elements() == 0) {
        tensorflow::DataType data_type =
            TFF_TRY(TensorFlowDataTypeFromDataType(array_pb.dtype()));
        return tensorflow::Tensor(data_type, shape);
      }
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", array_pb.kind_case()));
    }
  }
}

absl::StatusOr<federated_language::Array> ArrayContentFromTensor(
    const tensorflow::Tensor& tensor) {
  if (!tensorflow::DataTypeCanUseMemcpy(tensor.dtype())) {
    return absl::InvalidArgumentError(
        absl::StrCat("DataType ", tensorflow::DataTypeString(tensor.dtype()),
                     " cannot be serialized to raw Array content."));
  }
  federated_language::Array array_pb;
  array_pb.set_dtype(TFF_TRY(DataTypeFromTensorFlowDataType(tensor.dtype())));
  federated_language::ArrayShape shape_pb =
      TFF_TRY(ArrayShapeFromTensorShape(tensor.shape()));
  array_pb.mutable_shape()->Swap(&shape_pb);

  array_pb.set_content(tensor.tensor_data());

  return array_pb;
}

// Deserializes a `tensorflow::Tensor` directly from an Array proto containing
// raw `content`.
//
// Direct buffer construction via chunk streaming is used rather than
// converting to `tensorflow::TensorProto` and invoking `FromProto`. This design
// achieves several key goals:
// 1. Single-copy deserialization: Avoids allocating an intermediate
//    `std::string` or `TensorProto` buffer, copying bytes directly from the
//    `absl::Cord` chunks into the newly allocated `TensorBuffer`.
// 2. Cross-platform Protobuf compatibility: Open-source Protobuf does not
//    generate `mutable_<field>()` pointer accessors for `[ctype = CORD]`
//    fields, and `TensorProto.tensor_content` uses `std::string` in open-source
//    TensorFlow builds but `absl::Cord` in internal environments.
// 3. Abseil portability: Chunk-based iteration with `std::memcpy` provides
//    universal compatibility across all Abseil releases (including versions
//    prior to `absl::CopyCordToSpan` and environments where `CopyToArray` is
//    unavailable).
absl::StatusOr<tensorflow::Tensor> TensorFromArrayContent(
    const federated_language::Array& array_pb) {
  if (!array_pb.has_content()) {
    return absl::InvalidArgumentError("Expected a content field, found none.");
  }
  if (array_pb.kind_case() != federated_language::Array::KIND_NOT_SET) {
    return TensorFromArray(array_pb);
  }

  tensorflow::DataType data_type =
      TFF_TRY(TensorFlowDataTypeFromDataType(array_pb.dtype()));
  tensorflow::TensorShape shape =
      TFF_TRY(TensorShapeFromArrayShape(array_pb.shape()));

  if (shape.num_elements() == 0) {
    return tensorflow::Tensor(data_type, shape);
  }

  if (data_type == tensorflow::DT_STRING) {
    tensorflow::Tensor tensor(data_type, shape);
    if (array_pb.content().empty()) {
      return tensor;
    }
    tensorflow::tstring* strings = tensor.flat<tensorflow::tstring>().data();
    absl::Cord flat_cord = array_pb.content();
    absl::string_view reader = flat_cord.Flatten();
    std::vector<uint32_t> sizes(shape.num_elements());
    int64_t total_size = 0;
    for (uint32_t& size : sizes) {
      if (!tensorflow::core::GetVarint32(&reader, &size)) {
        return absl::InvalidArgumentError(
            "Failed to parse varint in serialized string tensor content.");
      }
      total_size += size;
    }
    if (total_size != static_cast<int64_t>(reader.size())) {
      return absl::InvalidArgumentError(
          "Serialized string tensor sizes do not match content size.");
    }
    for (int64_t i = 0; i < shape.num_elements(); ++i) {
      uint32_t size = sizes[i];
      strings[i].assign(reader.data(), size);
      reader.remove_prefix(size);
    }
    return tensor;
  }

  if (!tensorflow::DataTypeCanUseMemcpy(data_type)) {
    return absl::InvalidArgumentError(
        absl::StrCat("DataType ", tensorflow::DataTypeString(data_type),
                     " cannot be deserialized from raw Array content."));
  }

  int64_t expected_bytes =
      shape.num_elements() * tensorflow::DataTypeSize(data_type);
  if (array_pb.content().size() != expected_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Serialized tensor content size (", array_pb.content().size(),
        ") does not match expected (", expected_bytes, ") for shape ",
        shape.DebugString(), " and DataType ",
        tensorflow::DataTypeString(data_type), "."));
  }

  tensorflow::Tensor tensor(data_type, shape);
  char* dest = const_cast<char*>(tensor.tensor_data().data());
  for (absl::string_view chunk : array_pb.content().Chunks()) {
    std::memcpy(dest, chunk.data(), chunk.size());
    dest += chunk.size();
  }

  if (data_type == tensorflow::DT_BOOL) {
    const char* data = tensor.tensor_data().data();
    for (int64_t i = 0; i < shape.num_elements(); ++i) {
      if (data[i] != 0 && data[i] != 1) {
        return absl::InvalidArgumentError(
            "Invalid boolean byte found in serialized content.");
      }
    }
  }

  return tensor;
}

std::string GetNodeName(absl::string_view tensor_name) {
  absl::string_view::size_type pos = tensor_name.find(':');
  if (pos == absl::string_view::npos) {
    return std::string(tensor_name);
  } else {
    return std::string(tensor_name.substr(0, pos));
  }
}

}  // namespace tensorflow_federated
