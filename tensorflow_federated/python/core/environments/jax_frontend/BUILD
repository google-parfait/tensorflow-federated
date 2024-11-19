load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":jax_frontend_packages",
        "//tensorflow_federated/python/core/environments:environments_users",
        "//tensorflow_federated/python/core/environments/jax:jax_packages",
        "//tensorflow_federated/python/core/environments/xla_backend:xla_backend_packages",
    ],
)

package_group(
    name = "jax_frontend_packages",
    packages = ["//tensorflow_federated/python/core/environments/jax_frontend/..."],
)

licenses(["notice"])

py_library(
    name = "jax_frontend",
    srcs = ["__init__.py"],
    visibility = ["//tools/python_package:python_package_tool"],
)

py_library(
    name = "jax_computation",
    srcs = ["jax_computation.py"],
    deps = [
        ":jax_serialization",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "jax_computation_test",
    size = "small",
    srcs = ["jax_computation_test.py"],
    deps = [
        ":jax_computation",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "jax_computation_context",
    srcs = ["jax_computation_context.py"],
    deps = ["@federated_language//federated_language"],
)

py_test(
    name = "jax_computation_context_test",
    size = "small",
    srcs = ["jax_computation_context_test.py"],
    deps = [
        ":jax_computation",
        ":jax_computation_context",
    ],
)

py_library(
    name = "jax_serialization",
    srcs = ["jax_serialization.py"],
    deps = [
        ":jax_computation_context",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/xla_backend:xla_serialization",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_test(
    name = "jax_serialization_test",
    size = "small",
    srcs = ["jax_serialization_test.py"],
    deps = [
        ":jax_serialization",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/xla_backend:xla_serialization",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)
