load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    packages = [
        "//tensorflow_federated/experimental/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "learning",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow_federated/experimental:__pkg__",
        "//tensorflow_federated/tools:__subpackages__",
    ],
    deps = [":jax_components"],
)

py_library(
    name = "jax_components",
    srcs = ["jax_components.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/experimental/python/core/api:computations",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:iterative_process",
    ],
)

py_test(
    name = "jax_components_test",
    srcs = ["jax_components_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":jax_components",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/backends/xla:execution_contexts",
    ],
)
