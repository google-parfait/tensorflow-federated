package(default_visibility = [
    "//tensorflow_federated/python/core/api:__subpackages__",
    "//tensorflow_federated/python/core/backends:__subpackages__",
    "//tensorflow_federated/python/core/framework:__subpackages__",
    "//tensorflow_federated/python/core/impl/context_stack:__subpackages__",
    "//tensorflow_federated/python/core/impl/wrappers:__subpackages__",
    "//tensorflow_federated/python/core/templates:__subpackages__",
    "//tensorflow_federated/python/core/utils:__subpackages__",
    # TODO(b/143772942): This package is temporary and can be removed once all
    # the appropriate  modules have been moved out of the `impl` package.
    "//tensorflow_federated/python/core/impl:__pkg__",
])

licenses(["notice"])

py_library(
    name = "executors",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "caching_executor",
    srcs = ["caching_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

py_test(
    name = "caching_executor_test",
    size = "small",
    srcs = ["caching_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":caching_executor",
        ":eager_tf_executor",
        ":executor_base",
        ":reference_resolving_executor",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:executor_test_utils",
    ],
)

py_library(
    name = "cardinalities_utils",
    srcs = ["cardinalities_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
    ],
)

py_test(
    name = "cardinalities_utils_test",
    srcs = ["cardinalities_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cardinalities_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
    ],
)

py_library(
    name = "composing_executor",
    srcs = ["composing_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:type_utils",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_test(
    name = "composing_executor_test",
    size = "small",
    srcs = ["composing_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":caching_executor",
        ":composing_executor",
        ":eager_tf_executor",
        ":federating_executor",
        ":reference_resolving_executor",
        ":thread_delegating_executor",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "eager_tf_executor",
    srcs = ["eager_tf_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:serialization_utils",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:type_utils",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/tensorflow_libs:graph_merge",
    ],
)

py_test(
    name = "eager_tf_executor_test",
    size = "small",
    srcs = ["eager_tf_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:executor_test_utils",
    ],
)

py_library(
    name = "execution_context",
    srcs = ["execution_context.py"],
    srcs_version = "PY3",
    deps = [
        ":cardinalities_utils",
        ":executor_base",
        ":executor_factory",
        ":executor_value_base",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
        "//tensorflow_federated/python/core/impl:context_base",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

py_test(
    name = "execution_context_test",
    size = "small",
    srcs = ["execution_context_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":execution_context",
        ":executor_stacks",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "executor_base",
    srcs = ["executor_base.py"],
    srcs_version = "PY3",
    deps = [":executor_value_base"],
)

py_library(
    name = "executor_factory",
    srcs = ["executor_factory.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
    ],
)

py_test(
    name = "executor_factory_test",
    size = "small",
    srcs = ["executor_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":executor_base",
        ":executor_factory",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
    ],
)

py_library(
    name = "executor_service",
    srcs = ["executor_service.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl:executor_service_utils",
    ],
)

py_test(
    name = "executor_service_test",
    size = "small",
    srcs = ["executor_service_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":executor_base",
        ":executor_service",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl:executor_service_utils",
    ],
)

py_library(
    name = "executor_stacks",
    srcs = ["executor_stacks.py"],
    srcs_version = "PY3",
    deps = [
        ":caching_executor",
        ":composing_executor",
        ":eager_tf_executor",
        ":executor_base",
        ":executor_factory",
        ":federating_executor",
        ":reference_resolving_executor",
        ":sizing_executor",
        ":thread_delegating_executor",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
    ],
)

py_test(
    name = "executor_stacks_test",
    size = "small",
    srcs = ["executor_stacks_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        ":execution_context",
        ":executor_factory",
        ":executor_stacks",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl:executor_test_utils",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "executor_utils",
    srcs = ["executor_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:type_utils",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_test(
    name = "executor_utils_test",
    size = "small",
    srcs = ["executor_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":executor_utils"],
)

py_library(
    name = "executor_value_base",
    srcs = ["executor_value_base.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/core/api:typed_object"],
)

py_library(
    name = "federating_executor",
    srcs = ["federating_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:type_utils",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_test(
    name = "federating_executor_test",
    size = "small",
    srcs = ["federating_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":federating_executor",
        ":reference_resolving_executor",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:executor_test_utils",
        "//tensorflow_federated/python/core/impl/compiler:building_block_factory",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_library(
    name = "reference_resolving_executor",
    srcs = ["reference_resolving_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:type_utils",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:transformation_utils",
        "//tensorflow_federated/python/core/impl/compiler:type_serialization",
    ],
)

py_test(
    name = "reference_resolving_executor_test",
    size = "small",
    srcs = ["reference_resolving_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eager_tf_executor",
        ":federating_executor",
        ":reference_resolving_executor",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl:executor_test_utils",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)

py_library(
    name = "remote_executor",
    srcs = ["remote_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_utils",
        ":executor_value_base",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:executor_service_utils",
        "//tensorflow_federated/python/core/impl/executors:execution_context",
    ],
)

py_test(
    name = "remote_executor_test",
    size = "small",
    srcs = ["remote_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":executor_service",
        ":executor_stacks",
        ":reference_resolving_executor",
        ":remote_executor",
        "//tensorflow_federated/proto/v0:executor_py_pb2",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl:executor_test_utils",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/executors:execution_context",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "sizing_executor",
    srcs = ["sizing_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":executor_value_base",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:typed_object",
    ],
)

py_test(
    name = "sizing_executor_test",
    size = "small",
    srcs = ["sizing_executor_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":caching_executor",
        ":eager_tf_executor",
        ":reference_resolving_executor",
        ":sizing_executor",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_library(
    name = "thread_delegating_executor",
    srcs = ["thread_delegating_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        "//tensorflow_federated/python/common_libs:py_typecheck",
    ],
)

py_test(
    name = "thread_delegating_executor_test",
    size = "small",
    srcs = ["thread_delegating_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":caching_executor",
        ":eager_tf_executor",
        ":executor_base",
        ":thread_delegating_executor",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_library(
    name = "transforming_executor",
    srcs = ["transforming_executor.py"],
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl:type_utils",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
    ],
)

py_test(
    name = "transforming_executor_test",
    size = "small",
    srcs = ["transforming_executor_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":executor_base",
        ":transforming_executor",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/compiler:building_blocks",
        "//tensorflow_federated/python/core/impl/compiler:tree_transformations",
        "//tensorflow_federated/python/core/impl/compiler:type_factory",
    ],
)
