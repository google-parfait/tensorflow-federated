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
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "google/protobuf/repeated_ptr_field.h"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/tensor_coding.h"
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

#define TFF_ASSIGN_ARRAY_CASE(DTYPE, TYPE, FIELD)                         \
  case tensorflow::DataType::DTYPE: {                                     \
    const tensorflow::TTypes<TYPE>::ConstFlat flat = tensor.flat<TYPE>(); \
    array_pb.mutable_##FIELD()->mutable_value()->Assign(                  \
        flat.data(), flat.data() + flat.size());                          \
    break;                                                                \
  }

  switch (tensor.dtype()) {
    TFF_ASSIGN_ARRAY_CASE(DT_BOOL, bool, bool_list);
    TFF_ASSIGN_ARRAY_CASE(DT_INT8, int8_t, int8_list);
    TFF_ASSIGN_ARRAY_CASE(DT_INT16, int16_t, int16_list);
    TFF_ASSIGN_ARRAY_CASE(DT_INT32, int32_t, int32_list);
    TFF_ASSIGN_ARRAY_CASE(DT_INT64, int64_t, int64_list);
    TFF_ASSIGN_ARRAY_CASE(DT_UINT8, uint8_t, uint8_list);
    TFF_ASSIGN_ARRAY_CASE(DT_UINT16, uint16_t, uint16_list);
    TFF_ASSIGN_ARRAY_CASE(DT_UINT32, uint32_t, uint32_list);
    TFF_ASSIGN_ARRAY_CASE(DT_UINT64, uint64_t, uint64_list);
    TFF_ASSIGN_ARRAY_CASE(DT_FLOAT, float, float32_list);
    TFF_ASSIGN_ARRAY_CASE(DT_DOUBLE, double, float64_list);
#undef TFF_ASSIGN_ARRAY_CASE
    case tensorflow::DataType::DT_HALF: {
      const tensorflow::TTypes<Eigen::half>::ConstFlat flat =
          tensor.flat<Eigen::half>();
      google::protobuf::RepeatedField<int32_t>* list =
          array_pb.mutable_float16_list()->mutable_value();
      list->Reserve(flat.size());
      for (int64_t i = 0; i < flat.size(); ++i) {
        list->AddAlreadyReserved(static_cast<int32_t>(
            Eigen::numext::bit_cast<uint16_t>(flat.data()[i])));
      }
      break;
    }
    case tensorflow::DataType::DT_COMPLEX64: {
      const tensorflow::TTypes<tensorflow::complex64>::ConstFlat flat =
          tensor.flat<tensorflow::complex64>();
      const float* data = reinterpret_cast<const float*>(flat.data());
      array_pb.mutable_complex64_list()->mutable_value()->Assign(
          data, data + 2 * flat.size());
      break;
    }
    case tensorflow::DataType::DT_COMPLEX128: {
      const tensorflow::TTypes<tensorflow::complex128>::ConstFlat flat =
          tensor.flat<tensorflow::complex128>();
      const double* data = reinterpret_cast<const double*>(flat.data());
      array_pb.mutable_complex128_list()->mutable_value()->Assign(
          data, data + 2 * flat.size());
      break;
    }
    case tensorflow::DataType::DT_BFLOAT16: {
      const tensorflow::TTypes<Eigen::bfloat16>::ConstFlat flat =
          tensor.flat<Eigen::bfloat16>();
      google::protobuf::RepeatedField<int32_t>* list =
          array_pb.mutable_bfloat16_list()->mutable_value();
      list->Reserve(flat.size());
      for (int64_t i = 0; i < flat.size(); ++i) {
        list->AddAlreadyReserved(static_cast<int32_t>(
            Eigen::numext::bit_cast<uint16_t>(flat.data()[i])));
      }
      break;
    }
    case tensorflow::DataType::DT_STRING: {
      const tensorflow::TTypes<tensorflow::tstring>::ConstFlat flat =
          tensor.flat<tensorflow::tstring>();
      google::protobuf::RepeatedPtrField<std::string>* list =
          array_pb.mutable_string_list()->mutable_value();
      list->Reserve(flat.size());
      for (int64_t i = 0; i < flat.size(); ++i) {
        list->Add(std::string(flat.data()[i]));
      }
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", tensor.dtype()));
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
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unexpected DataType found:", array_pb.kind_case()));
  }
}

static absl::Cord TensorDataToCord(const tensorflow::Tensor& tensor) {
  if (tensor.TotalBytes() == 0) {
    return absl::Cord();
  }
  return absl::MakeCordFromExternal(tensor.tensor_data(),
                                    [t = tensor](absl::string_view) {});
}

absl::StatusOr<federated_language::Array> ArrayContentFromTensor(
    const tensorflow::Tensor& tensor) {
  federated_language::Array array_pb;
  federated_language::DataType data_type =
      TFF_TRY(DataTypeFromTensorFlowDataType(tensor.dtype()));
  array_pb.set_dtype(data_type);
  federated_language::ArrayShape shape_pb =
      TFF_TRY(ArrayShapeFromTensorShape(tensor.shape()));
  array_pb.mutable_shape()->Swap(&shape_pb);

  if (tensorflow::DataTypeCanUseMemcpy(tensor.dtype())) {
    array_pb.set_content(TensorDataToCord(tensor));
  } else if (tensor.dtype() == tensorflow::DT_STRING) {
#if defined(TENSORFLOW_PROTOBUF_USES_CORD)
    absl::Cord cord;
    tensorflow::port::EncodeStringList(
        tensor.flat<tensorflow::tstring>().data(), tensor.NumElements(), &cord);
    array_pb.set_content(std::move(cord));
#else
    std::string str;
    tensorflow::port::EncodeStringList(
        tensor.flat<tensorflow::tstring>().data(), tensor.NumElements(), &str);
    array_pb.set_content(std::move(str));
#endif
  } else {
    return absl::UnimplementedError(absl::StrCat(
        "ArrayContentFromTensor does not support DataType: ", tensor.dtype()));
  }

  return array_pb;
}

absl::StatusOr<tensorflow::Tensor> TensorFromArrayContent(
    const federated_language::Array& array_pb) {
  if (!array_pb.has_content()) {
    return absl::InvalidArgumentError("Expected a content field, found none.");
  }

  tensorflow::DataType data_type =
      TFF_TRY(TensorFlowDataTypeFromDataType(array_pb.dtype()));
  tensorflow::TensorShape shape =
      TFF_TRY(TensorShapeFromArrayShape(array_pb.shape()));

  tensorflow::Tensor tensor(data_type, shape);
  if (tensorflow::DataTypeCanUseMemcpy(data_type)) {
    if (array_pb.content().size() != tensor.TotalBytes()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unexpected content size for tensor: expected ", tensor.TotalBytes(),
          ", found ", array_pb.content().size()));
    }
    if (tensor.TotalBytes() > 0) {
      char* dst = static_cast<char*>(tensor.data());
      for (absl::string_view chunk : array_pb.content().Chunks()) {
        std::memcpy(dst, chunk.data(), chunk.size());
        dst += chunk.size();
      }
    }
  } else if (data_type == tensorflow::DT_STRING) {
    if (array_pb.content().empty()) {
      return tensor;
    }
#if defined(TENSORFLOW_PROTOBUF_USES_CORD)
    if (!tensorflow::port::DecodeStringList(
            array_pb.content(), tensor.flat<tensorflow::tstring>().data(),
            tensor.NumElements())) {
      return absl::InvalidArgumentError(
          "Serialized string tensor could not be decoded.");
    }
#else
    if (!tensorflow::port::DecodeStringList(
            std::string(array_pb.content()),
            tensor.flat<tensorflow::tstring>().data(), tensor.NumElements())) {
      return absl::InvalidArgumentError(
          "Serialized string tensor could not be decoded.");
    }
#endif
  } else {
    return absl::UnimplementedError(absl::StrCat(
        "TensorFromArrayContent does not support DataType: ", data_type));
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
