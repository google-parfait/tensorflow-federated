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
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/core/api:computation_base",
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
    name = "func_utils",
    srcs = ["func_utils.py"],
    deps = [
        ":anonymous_tuple",
        ":type_utils",
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
    name = "serialization",
    srcs = ["serialization.py"],
    deps = [
        ":graph_utils",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/core/api:types",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    deps = [
        ":serialization",
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
    name = "type_utils",
    srcs = ["type_utils.py"],
    deps = [
        ":anonymous_tuple",
        "//tensorflow_federated/python/core/api:types",
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
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
