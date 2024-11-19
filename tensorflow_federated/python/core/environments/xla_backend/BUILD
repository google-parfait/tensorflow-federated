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
    visibility = ["//tools/python_package:python_package_tool"],
)

py_library(
    name = "xla_executor_bindings",
    srcs = ["xla_executor_bindings.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = ["//tensorflow_federated/cc/core/impl/executors:executor_bindings"],
)

py_test(
    name = "xla_executor_bindings_test",
    srcs = ["xla_executor_bindings_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [":xla_executor_bindings"],
)

py_library(
    name = "xla_serialization",
    srcs = ["xla_serialization.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "@com_google_protobuf//:protobuf_python",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_test(
    name = "xla_serialization_test",
    size = "small",
    srcs = ["xla_serialization_test.py"],
    deps = [
        ":xla_serialization",
        "@com_google_protobuf//:protobuf_python",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)
