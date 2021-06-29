load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core/impl:impl_visibility"],
    packages = [
        "//tensorflow_federated/python/core/impl/executor_stacks/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "executor_stacks",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "executor_stack_bindings",
    srcs = ["executor_stack_bindings.py"],
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
        "//tensorflow_federated/cc/core/impl/executor_stacks:executor_stack_bindings",
        "//tensorflow_federated/python/core/impl/bindings_utils:data_conversions",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)

py_test(
    name = "executor_stack_bindings_test",
    srcs = ["executor_stack_bindings_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
        ":executor_stack_bindings",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
        "//tensorflow_federated/python/core/impl/types:placements",
    ],
)
