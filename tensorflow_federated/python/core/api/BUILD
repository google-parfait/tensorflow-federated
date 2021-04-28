load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core:core_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/api/...",

        # Core Dependencies
        "//tensorflow_federated/python/core/backends/...",
        "//tensorflow_federated/python/core/templates/...",
        "//tensorflow_federated/python/core/utils/...",
        "//tensorflow_federated/python/core/test/...",
        "//tensorflow_federated/python/core/types/...",

        # Namespace Dependencies
        "//tensorflow_federated",

        # TODO(b/181569610): Clean up circular dependency between the `api` and
        # `impl` packages.
        "//tensorflow_federated/python/core/impl/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "api",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "computation_base",
    srcs = ["computation_base.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/types:typed_object",
    ],
)

py_library(
    name = "computations",
    srcs = ["computations.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/impl/wrappers:computation_wrapper_instances"],
)

py_library(
    name = "test_case",
    srcs = ["test_case.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "@absl_py//absl/testing:absltest",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "test_case_test",
    size = "small",
    srcs = ["test_case_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":test_case",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)
