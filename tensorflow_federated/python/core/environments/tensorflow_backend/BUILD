load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//tools:build_defs.bzl", "py_cpu_gpu_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":tensorflow_backend_packages",
        "//tensorflow_federated/python/core/environments:environments_users",
        "//tensorflow_federated/python/core/environments/tensorflow:tensorflow_packages",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_frontend_packages",
    ],
)

package_group(
    name = "tensorflow_backend_packages",
    packages = ["//tensorflow_federated/python/core/environments/tensorflow_backend/..."],
)

licenses(["notice"])

py_library(
    name = "tensorflow_backend",
    srcs = ["__init__.py"],
    visibility = ["//tools/python_package:python_package_tool"],
)

py_library(
    name = "compiled_computation_transformations",
    srcs = ["compiled_computation_transformations.py"],
    deps = [
        ":graph_optimizations",
        ":graph_spec",
        ":serialization_utils",
        ":tensorflow_computation_transformations",
        ":tensorflow_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_test(
    name = "compiled_computation_transformations_test",
    size = "small",
    srcs = ["compiled_computation_transformations_test.py"],
    deps = [
        ":compiled_computation_transformations",
        ":tensorflow_computation_factory",
        ":tensorflow_computation_test_utils",
        ":tensorflow_computation_transformations",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "graph_optimizations",
    srcs = ["graph_optimizations.py"],
    deps = [":graph_spec"],
)

py_test(
    name = "graph_optimizations_test",
    srcs = ["graph_optimizations_test.py"],
    deps = [
        ":graph_optimizations",
        ":graph_spec",
    ],
)

py_library(
    name = "graph_spec",
    srcs = ["graph_spec.py"],
)

py_test(
    name = "graph_spec_test",
    srcs = ["graph_spec_test.py"],
    deps = [":graph_spec"],
)

py_library(
    name = "graph_utils",
    srcs = ["graph_utils.py"],
)

py_test(
    name = "graph_utils_test",
    srcs = ["graph_utils_test.py"],
    deps = [":graph_utils"],
)

py_library(
    name = "serialization_utils",
    srcs = ["serialization_utils.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "serialization_utils_test",
    size = "small",
    srcs = ["serialization_utils_test.py"],
    deps = [
        ":serialization_utils",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "tensorflow_building_block_factory",
    srcs = ["tensorflow_building_block_factory.py"],
    deps = [
        ":tensorflow_computation_factory",
        ":type_conversions",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "tensorflow_building_block_factory_test",
    srcs = ["tensorflow_building_block_factory_test.py"],
    deps = [
        ":tensorflow_building_block_factory",
        ":tensorflow_computation_test_utils",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "tensorflow_computation_factory",
    srcs = ["tensorflow_computation_factory.py"],
    deps = [
        ":serialization_utils",
        ":tensorflow_utils",
        ":type_conversions",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_cpu_gpu_test(
    name = "tensorflow_computation_factory_test",
    srcs = ["tensorflow_computation_factory_test.py"],
    deps = [
        ":tensorflow_computation_factory",
        ":tensorflow_computation_test_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_library(
    name = "tensorflow_computation_test_utils",
    testonly = True,
    srcs = ["tensorflow_computation_test_utils.py"],
    deps = [
        ":tensorflow_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_library(
    name = "tensorflow_computation_transformations",
    srcs = ["tensorflow_computation_transformations.py"],
    deps = [
        ":serialization_utils",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_test(
    name = "tensorflow_computation_transformations_test",
    size = "small",
    srcs = ["tensorflow_computation_transformations_test.py"],
    deps = [
        ":serialization_utils",
        ":tensorflow_computation_factory",
        ":tensorflow_computation_transformations",
        ":tensorflow_utils",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_library(
    name = "tensorflow_executor_bindings",
    srcs = ["tensorflow_executor_bindings.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = ["//tensorflow_federated/cc/core/impl/executors:executor_bindings"],
)

py_test(
    name = "tensorflow_executor_bindings_test",
    srcs = ["tensorflow_executor_bindings_test.py"],
    tags = [
        "nokokoro",  # b/193543632: C++ execution is not fully supported in OSS.
    ],
    deps = [
        ":tensorflow_computation_factory",
        ":tensorflow_executor_bindings",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "//tensorflow_federated/python/core/impl/executors:executor_bindings",
        "//tensorflow_federated/python/core/impl/executors:value_serialization",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "tensorflow_test_utils",
    testonly = True,
    srcs = ["tensorflow_test_utils.py"],
)

py_library(
    name = "tensorflow_tree_transformations",
    srcs = ["tensorflow_tree_transformations.py"],
    deps = [
        ":tensorflow_building_block_factory",
        ":tensorflow_computation_factory",
        ":type_conversions",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "tensorflow_tree_transformations_test",
    srcs = ["tensorflow_tree_transformations_test.py"],
    deps = [
        ":tensorflow_tree_transformations",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "tensorflow_utils",
    srcs = ["tensorflow_utils.py"],
    deps = [
        ":graph_utils",
        ":serialization_utils",
        ":type_conversions",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_test(
    name = "tensorflow_utils_test",
    size = "small",
    srcs = ["tensorflow_utils_test.py"],
    deps = [
        ":serialization_utils",
        ":tensorflow_test_utils",
        ":tensorflow_utils",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "@federated_language//federated_language",
        "@federated_language//federated_language/proto:computation_py_pb2",
    ],
)

py_library(
    name = "type_conversions",
    srcs = ["type_conversions.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "type_conversions_test",
    size = "small",
    srcs = ["type_conversions_test.py"],
    deps = [
        ":type_conversions",
        "@federated_language//federated_language",
    ],
)
