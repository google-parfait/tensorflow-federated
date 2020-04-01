package(default_visibility = [
    # Subpackages
    "//tensorflow_federated/python/core/utils:__subpackages__",
])

licenses(["notice"])

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow_federated:__pkg__",
        "//tensorflow_federated/python/core:__pkg__",
    ],
    deps = [
        ":computation_utils",
        ":differential_privacy",
        ":encoding_utils",
        ":execution_tracing",
        ":federated_aggregations",
        ":tf_computation_utils",
    ],
)

py_library(
    name = "computation_utils",
    srcs = ["computation_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:value_base",
        "//tensorflow_federated/python/core/api:values",
    ],
)

py_test(
    name = "computation_utils_test",
    size = "small",
    srcs = ["computation_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/executors:default_executor",
        "//tensorflow_federated/python/core/impl/executors:executor_stacks",
    ],
)

py_library(
    name = "differential_privacy",
    srcs = ["differential_privacy.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

py_test(
    name = "differential_privacy_test",
    srcs = ["differential_privacy_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":differential_privacy",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl:type_utils",
        "//tensorflow_federated/python/core/impl/compiler:placement_literals",
        "//tensorflow_federated/python/core/impl/executors:default_executor",
    ],
)

py_library(
    name = "encoding_utils",
    srcs = ["encoding_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":computation_utils",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl:type_utils",
    ],
)

py_test(
    name = "encoding_utils_test",
    srcs = ["encoding_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":computation_utils",
        ":encoding_utils",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/executors:default_executor",
    ],
)

py_library(
    name = "execution_tracing",
    srcs = ["execution_tracing.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:tracing",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/impl:computation_impl",
        "//tensorflow_federated/python/core/impl/compiler:intrinsic_defs",
        "//tensorflow_federated/python/core/impl/executors:caching_executor",
        "//tensorflow_federated/python/core/impl/executors:composing_executor",
        "//tensorflow_federated/python/core/impl/executors:eager_tf_executor",
        "//tensorflow_federated/python/core/impl/executors:federating_executor",
        "//tensorflow_federated/python/core/impl/executors:reference_resolving_executor",
        "//tensorflow_federated/python/core/impl/executors:remote_executor",
        "//tensorflow_federated/python/core/impl/executors:sizing_executor",
        "//tensorflow_federated/python/core/impl/executors:thread_delegating_executor",
        "//tensorflow_federated/python/core/impl/executors:transforming_executor",
    ],
)

py_test(
    name = "execution_tracing_test",
    size = "small",
    srcs = ["execution_tracing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":execution_tracing",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/common_libs:tracing",
    ],
)

py_library(
    name = "federated_aggregations",
    srcs = ["federated_aggregations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/api:value_base",
    ],
)

py_test(
    name = "federated_aggregations_test",
    size = "small",
    srcs = ["federated_aggregations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_aggregations",
        "//tensorflow_federated/python/common_libs:test",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:placements",
        "//tensorflow_federated/python/core/impl/executors:default_executor",
    ],
)

py_library(
    name = "tf_computation_utils",
    srcs = ["tf_computation_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_types",
    ],
)

py_test(
    name = "tf_computation_utils_test",
    size = "small",
    srcs = ["tf_computation_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tf_computation_utils",
        "//tensorflow_federated/python/common_libs:anonymous_tuple",
        "//tensorflow_federated/python/common_libs:test",
    ],
)
