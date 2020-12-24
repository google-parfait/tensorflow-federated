load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    packages = ["//tensorflow_federated/experimental/python/core/impl/jax_context/..."],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "jax_context",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "jax_computation_context",
    srcs = ["jax_computation_context.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/context_stack:context_base"],
)

py_test(
    name = "jax_computation_context_test",
    size = "small",
    srcs = ["jax_computation_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":jax_computation_context",
        "//tensorflow_federated/python/core/api:computations",
    ],
)
