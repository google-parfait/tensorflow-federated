load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":xla_backend_packages",
        "//tensorflow_federated/python/core/environments:environments_users",
        "//tensorflow_federated/python/core/environments/jax:jax_packages",
        "//tensorflow_federated/python/core/environments/jax_frontend:jax_frontend_packages",
    ],
)

package_group(
    name = "xla_backend_packages",
    packages = ["//tensorflow_federated/python/core/environments/xla_backend/..."],
)

licenses(["notice"])

py_library(
    name = "xla_backend",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "compiler",
    srcs = ["compiler.py"],
    deps = [
        ":xla_serialization",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/compiler:local_computation_factory_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//third_party/jax",
        "@pypi_numpy//:pkg",
    ],
)

py_test(
    name = "compiler_test",
    size = "small",
    srcs = ["compiler_test.py"],
    deps = [
        ":compiler",
        ":runtime",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//third_party/jax",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "runtime",
    srcs = ["runtime.py"],
    deps = [
        ":xla_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:array_shape",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:typed_object",
        "//third_party/jax",
        "@pypi_numpy//:pkg",
    ],
)

py_test(
    name = "runtime_test",
    srcs = ["runtime_test.py"],
    deps = [
        ":runtime",
        ":xla_serialization",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//third_party/jax",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)

py_library(
    name = "xla_executor_bindings",
    srcs = ["xla_executor_bindings.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        "//tensorflow_federated/cc/core/impl/executors:executor_bindings",
        "@pypi_tensorflow//:pkg",  # TODO: b/305743962 - Remove the tensorflow dependency.
    ],
)

py_test(
    name = "xla_executor_bindings_test",
    srcs = ["xla_executor_bindings_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":xla_executor_bindings",
        "@pypi_absl_py//:pkg",
    ],
)

py_library(
    name = "xla_serialization",
    srcs = ["xla_serialization.py"],
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//third_party/jax",
        "@com_google_protobuf//:protobuf_python",
        "@pypi_numpy//:pkg",
    ],
)

py_test(
    name = "xla_serialization_test",
    size = "small",
    srcs = ["xla_serialization_test.py"],
    deps = [
        ":xla_serialization",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//third_party/jax",
        "@com_google_protobuf//:protobuf_python",
        "@pypi_absl_py//:pkg",
        "@pypi_numpy//:pkg",
    ],
)
