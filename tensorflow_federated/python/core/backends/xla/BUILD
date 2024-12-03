load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":xla_packages",
        "//tensorflow_federated/python/core:core_users",
    ],
)

package_group(
    name = "xla_packages",
    packages = ["//tensorflow_federated/python/core/backends/xla/..."],
)

licenses(["notice"])

py_library(
    name = "xla",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/core/backends:__pkg__"],
)

py_library(
    name = "cpp_execution_contexts",
    srcs = ["cpp_execution_contexts.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        "//tensorflow_federated/python/core/backends/native:compiler",
        "//tensorflow_federated/python/core/environments/jax_frontend:jax_computation",
        "//tensorflow_federated/python/core/environments/xla_backend:xla_executor_bindings",
        "//tensorflow_federated/python/core/impl/executor_stacks:cpp_executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "cpp_execution_contexts_test",
    srcs = ["cpp_execution_contexts_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":cpp_execution_contexts",
        "//tensorflow_federated/python/core/environments/jax_frontend:jax_computation",
        "@federated_language//federated_language",
    ],
)
