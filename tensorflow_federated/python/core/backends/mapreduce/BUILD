load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":mapreduce_packages",
        "//tensorflow_federated/python/core:core_users",

        # TODO: b/233800075 - Remove dependency from `native` to `mapreduce`.
        "//tensorflow_federated/python/core/backends/native:native_packages",

        # TODO: b/349685709 - Remove dependency from `test` to `mapreduce`.
        "//tensorflow_federated/python/core/backends/test:test_packages",
    ],
)

package_group(
    name = "mapreduce_packages",
    packages = ["//tensorflow_federated/python/core/backends/mapreduce/..."],
)

licenses(["notice"])

py_library(
    name = "mapreduce",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/core/backends:__pkg__"],
    deps = [
        ":compiler",
        ":form_utils",
        ":forms",
        ":intrinsics",
    ],
)

py_library(
    name = "compiler",
    srcs = ["compiler.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:compiled_computation_transformations",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_tree_transformations",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/impl/compiler:transformations",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "compiler_test",
    srcs = ["compiler_test.py"],
    deps = [
        ":compiler",
        ":form_utils",
        ":mapreduce_test_utils",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_computation_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_block_test_utils",
        "//tensorflow_federated/python/core/impl/executor_stacks:executor_factory",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_library(
    name = "forms",
    srcs = ["forms.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "forms_test",
    srcs = ["forms_test.py"],
    deps = [
        ":distribute_aggregate_test_utils",
        ":forms",
        ":mapreduce_test_utils",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "form_utils",
    srcs = ["form_utils.py"],
    deps = [
        ":compiler",
        ":forms",
        ":intrinsics",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_building_block_factory",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_tree_transformations",
        "//tensorflow_federated/python/core/impl/compiler:transformations",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "form_utils_test",
    size = "large",
    srcs = ["form_utils_test.py"],
    deps = [
        ":distribute_aggregate_test_utils",
        ":form_utils",
        ":forms",
        ":intrinsics",
        ":mapreduce_test_utils",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:serialization_utils",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "distribute_aggregate_test_utils",
    testonly = True,
    srcs = ["distribute_aggregate_test_utils.py"],
    deps = [
        ":forms",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "intrinsics",
    srcs = ["intrinsics.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_computation_factory",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "intrinsics_test",
    srcs = ["intrinsics_test.py"],
    args = [
        "--golden",
        "$(location intrinsics_test_goldens/federated_secure_modular_sum.expected)",
    ],
    data = [
        "intrinsics_test_goldens/federated_secure_modular_sum.expected",
    ],
    deps = [
        ":intrinsics",
        "//tensorflow_federated/python/common_libs:golden",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "mapreduce_test_utils",
    testonly = True,
    srcs = ["mapreduce_test_utils.py"],
    deps = [
        ":forms",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "@federated_language//federated_language",
    ],
)
