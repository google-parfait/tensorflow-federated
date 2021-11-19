load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":xla_context_packages",
    "//tensorflow_federated/experimental/python/core/impl/jax_context:jax_context_packages",
    "//tensorflow_federated/python/core/impl:impl_visibility",
    "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
    "//tensorflow_federated/python/core/impl/executor_stacks:executor_stacks_packages",
    "//tensorflow_federated/python/core/impl/executors:executors_packages",
    "//tensorflow_federated/python/core/impl/wrappers:wrappers_packages",
])

package_group(
    name = "xla_context_packages",
    packages = ["//tensorflow_federated/python/core/impl/xla_context/..."],
)

licenses(["notice"])

py_library(
    name = "xla_context",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "xla_serialization",
    srcs = ["xla_serialization.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/types:computation_types",
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
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "@absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)
