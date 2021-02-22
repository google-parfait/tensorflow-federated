load("@rules_python//python:defs.bzl", "py_library", "py_test")

# Packages that have visibility into the `impl` package.
package_group(
    name = "impl_visibility",
    includes = [
        "//tensorflow_federated/python/core:core_visibility",
    ],
    packages = [
        "//tensorflow_federated/python/core/api/...",
        "//tensorflow_federated/python/core/backends/...",
        "//tensorflow_federated/python/core/framework/...",
        "//tensorflow_federated/python/core/templates/...",
        "//tensorflow_federated/python/core/test/...",
        "//tensorflow_federated/python/core/utils/...",
    ],
)

package_group(
    name = "default_visibility",
    includes = [":impl_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/core/impl/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "impl",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "tree_to_cc_transformations",
    srcs = ["tree_to_cc_transformations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:compiled_computation_transforms",
    ],
)

py_test(
    name = "tree_to_cc_transformations_test",
    size = "small",
    srcs = ["tree_to_cc_transformations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tree_to_cc_transformations",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/context_stack:set_default_context",
        "//tensorflow_federated/python/core/impl/executors:execution_context",
        "//tensorflow_federated/python/core/impl/executors:executor_stacks",
        "//tensorflow_federated/python/core/impl/wrappers:computation_wrapper_instances",
    ],
)
