load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    packages = ["//tensorflow_federated/experimental/python/core/impl/jax_context/..."],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "xla_serialization",
    srcs = ["xla_serialization.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "xla_serialization_test",
    size = "small",
    srcs = ["xla_serialization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":xla_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "@com_google_protobuf//:protobuf_python",
    ],
)
