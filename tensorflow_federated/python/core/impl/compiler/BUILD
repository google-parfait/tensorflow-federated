load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":compiler_packages",
        "//tensorflow_federated/python/core/impl:impl_users",
        "//tensorflow_federated/python/core/impl/execution_contexts:execution_contexts_packages",
        "//tensorflow_federated/python/core/impl/executors:executors_packages",
    ],
)

package_group(
    name = "compiler_packages",
    packages = ["//tensorflow_federated/python/core/impl/compiler/..."],
)

licenses(["notice"])

py_library(
    name = "compiler",
    srcs = ["__init__.py"],
    visibility = ["//tools/python_package:python_package_tool"],
)

py_library(
    name = "building_block_test_utils",
    testonly = True,
    srcs = ["building_block_test_utils.py"],
    deps = [
        "@com_google_protobuf//:protobuf_python",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "transformations",
    srcs = ["transformations.py"],
    deps = [
        ":tree_transformations",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "transformations_test",
    srcs = ["transformations_test.py"],
    deps = [
        ":building_block_test_utils",
        ":transformations",
        ":tree_transformations",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_library(
    name = "tree_transformations",
    srcs = ["tree_transformations.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "tree_transformations_test",
    size = "small",
    srcs = ["tree_transformations_test.py"],
    args = [
        "--golden",
        "$(location tree_transformations_test_goldens/uniquify_names_blocks_nested_inside_of_locals.expected)",
    ],
    data = [
        "tree_transformations_test_goldens/uniquify_names_blocks_nested_inside_of_locals.expected",
    ],
    deps = [
        ":building_block_test_utils",
        ":tree_transformations",
        "//tensorflow_federated/python/common_libs:golden",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "compiler_test_utils",
    testonly = True,
    srcs = ["compiler_test_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:golden",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "compiler_test_utils_test",
    size = "small",
    srcs = ["compiler_test_utils_test.py"],
    args = [
        "--golden",
        "$(location compiler_test_utils_test_goldens/test_check_computations.expected)",
    ],
    data = [
        "compiler_test_utils_test_goldens/test_check_computations.expected",
    ],
    deps = [
        ":building_block_test_utils",
        ":compiler_test_utils",
    ],
)
