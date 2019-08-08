package(default_visibility = [
    "//tensorflow_federated/python/core/api:__pkg__",
    "//tensorflow_federated/python/core/framework:__pkg__",
    "//tensorflow_federated/python/core/impl:__pkg__",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "compiler",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "building_blocks",
    srcs = ["building_blocks.py"],
    deps = [
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/impl:intrinsic_defs",
        "//tensorflow_federated/python/core/impl:placement_literals",
        "//tensorflow_federated/python/core/impl:type_serialization",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

py_test(
    name = "building_blocks_test",
    size = "small",
    srcs = ["building_blocks_test.py"],
    deps = [
        ":building_blocks",
        "//tensorflow_federated/proto/v0:tensorflow_federated_v0_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl:computation_constructing_utils",
        "//tensorflow_federated/python/core/impl:computation_test_utils",
        "//tensorflow_federated/python/core/impl:intrinsic_defs",
        "//tensorflow_federated/python/core/impl:type_serialization",
    ],
)
