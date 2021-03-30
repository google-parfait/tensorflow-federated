load("@rules_python//python:defs.bzl", "py_library", "py_test")

package_group(
    name = "default_visibility",
    includes = ["//tensorflow_federated/python/core:core_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/test/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "test",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":static_assert",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/context_stack:set_default_context",
        "//tensorflow_federated/python/simulation/datasets:from_tensor_slices_client_data",
    ],
)

py_library(
    name = "static_assert",
    srcs = ["static_assert.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tree_analysis",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
    ],
)

py_test(
    name = "static_assert_test",
    size = "small",
    srcs = ["static_assert_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":static_assert",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
    ],
)
