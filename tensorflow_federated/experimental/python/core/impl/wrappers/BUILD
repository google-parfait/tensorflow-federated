load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    packages = [
        "//tensorflow_federated/experimental/python/core/api/...",
        "//tensorflow_federated/experimental/python/core/impl/wrappers/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "wrappers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "computation_wrapper_instances",
    srcs = ["computation_wrapper_instances.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/experimental/python/core/impl/jax_context:jax_serialization",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/utils:function_utils",
        "//tensorflow_federated/python/core/impl/wrappers:computation_wrapper",
    ],
)

py_test(
    name = "computation_wrapper_instances_test",
    size = "small",
    srcs = ["computation_wrapper_instances_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_wrapper_instances",
        "//tensorflow_federated/python/core/impl:computation_impl",
    ],
)
