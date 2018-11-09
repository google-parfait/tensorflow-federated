package(default_visibility = ["//tensorflow_federated/python/core/api:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "anonymous_tuple",
    srcs = ["anonymous_tuple.py"],
    deps = ["@six_archive//:six"],
)

py_test(
    name = "anonymous_tuple_test",
    size = "small",
    srcs = ["anonymous_tuple_test.py"],
    deps = [":anonymous_tuple"],
)

py_library(
    name = "computation_impl",
    srcs = ["computation_impl.py"],
    deps = [
        ":context_stack",
        ":func_utils",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
    ],
)

py_test(
    name = "computation_test",
    size = "small",
    srcs = ["computation_test.py"],
    deps = [
        ":computation_impl",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
    ],
)

py_library(
    name = "computation_wrapper",
    srcs = ["computation_wrapper.py"],
    deps = [
        ":func_utils",
        "//tensorflow_federated/python/core/api:types",
    ],
)

py_library(
    name = "computation_wrapper_instances",
    srcs = ["computation_wrapper_instances.py"],
    deps = [
        ":computation_impl",
        ":computation_wrapper",
        ":tensorflow_serialization",
    ],
)

py_test(
    name = "computation_wrapper_instances_test",
    size = "small",
    srcs = ["computation_wrapper_instances_test.py"],
    deps = [
        ":computation_wrapper_instances",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "computation_wrapper_test",
    size = "small",
    srcs = ["computation_wrapper_test.py"],
    deps = [
        ":computation_wrapper",
        ":type_utils",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "context_base",
    srcs = ["context_base.py"],
)

py_library(
    name = "context_stack",
    srcs = ["context_stack.py"],
    deps = [
        ":context_base",
        ":default_context",
    ],
)

py_test(
    name = "context_stack_test",
    size = "small",
    srcs = ["context_stack_test.py"],
    deps = [
        ":context_base",
        ":context_stack",
        ":default_context",
    ],
)

py_library(
    name = "default_context",
    srcs = ["default_context.py"],
    deps = [":context_base"],
)

py_library(
    name = "func_utils",
    srcs = ["func_utils.py"],
    deps = [
        ":anonymous_tuple",
        ":type_utils",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:types",
    ],
)

py_test(
    name = "func_utils_test",
    size = "small",
    srcs = ["func_utils_test.py"],
    deps = [
        ":anonymous_tuple",
        ":func_utils",
        "//tensorflow_federated/python/core/api:types",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "graph_utils",
    srcs = ["graph_utils.py"],
    deps = [
        ":anonymous_tuple",
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "graph_utils_test",
    size = "small",
    srcs = ["graph_utils_test.py"],
    deps = [
        ":anonymous_tuple",
        ":graph_utils",
        ":test_utils",
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "tensorflow_serialization",
    srcs = ["tensorflow_serialization.py"],
    deps = [
        ":graph_utils",
        ":type_serialization",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tensorflow_serialization_test",
    size = "small",
    srcs = ["tensorflow_serialization_test.py"],
    deps = [
        ":tensorflow_serialization",
        ":type_serialization",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)

py_test(
    name = "test_utils_test",
    size = "small",
    srcs = ["test_utils_test.py"],
    deps = [
        ":test_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "type_serialization",
    srcs = ["type_serialization.py"],
    deps = [
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "type_serialization_test",
    size = "small",
    srcs = ["type_serialization_test.py"],
    deps = [
        ":type_serialization",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "type_utils",
    srcs = ["type_utils.py"],
    deps = [
        ":anonymous_tuple",
        "//tensorflow_federated/python/core/api:types",
        "//tensorflow_federated/python/core/api:value_base",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "type_utils_test",
    size = "small",
    srcs = ["type_utils_test.py"],
    deps = [
        ":anonymous_tuple",
        ":test_utils",
        ":type_utils",
        ":values",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "values",
    srcs = ["values.py"],
    deps = [
        ":anonymous_tuple",
        ":func_utils",
        ":type_serialization",
        ":type_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/core/api:types",
        "//tensorflow_federated/python/core/api:value_base",
        "@six_archive//:six",
    ],
)

py_test(
    name = "values_test",
    size = "small",
    srcs = ["values_test.py"],
    deps = [
        ":anonymous_tuple",
        ":tensorflow_serialization",
        ":values",
        "//tensorflow_federated/python/core/api:types",
        "//tensorflow_federated/python/core/api:value_base",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
