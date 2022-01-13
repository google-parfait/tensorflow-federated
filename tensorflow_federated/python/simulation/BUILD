load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":simulation_packages",
    ":simulation_visibility",
])

package_group(
    name = "simulation_packages",
    packages = ["//tensorflow_federated/python/simulation/..."],
)

package_group(
    name = "simulation_visibility",
    includes = ["//tensorflow_federated/tools:tools_packages"],
)

licenses(["notice"])

py_library(
    name = "simulation",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":iterative_process_compositions",
        ":sampling_utils",
        ":server_utils",
        ":training_loop",
        "//tensorflow_federated/python/simulation/baselines",
        "//tensorflow_federated/python/simulation/datasets",
        "//tensorflow_federated/python/simulation/models",
    ],
)

py_library(
    name = "iterative_process_compositions",
    srcs = ["iterative_process_compositions.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/templates:iterative_process",
    ],
)

py_test(
    name = "iterative_process_compositions_test",
    size = "small",
    srcs = ["iterative_process_compositions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":iterative_process_compositions",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:iterative_process",
    ],
)

py_library(
    name = "sampling_utils",
    srcs = ["sampling_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "sampling_utils_test",
    size = "small",
    srcs = ["sampling_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":sampling_utils"],
)

py_library(
    name = "server_utils",
    srcs = ["server_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/proto/v0:executor_py_pb2_grpc",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/executors:executor_factory",
        "//tensorflow_federated/python/core/impl/executors:executor_service",
    ],
)

py_test(
    name = "server_utils_test",
    size = "small",
    srcs = ["server_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":server_utils",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/executors:eager_tf_executor",
        "//tensorflow_federated/python/core/impl/executors:executor_stacks",
    ],
)

py_library(
    name = "training_loop",
    srcs = ["training_loop.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/program:program_state_manager",
        "//tensorflow_federated/python/program:release_manager",
    ],
)

py_test(
    name = "training_loop_test",
    size = "medium",
    srcs = ["training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":training_loop",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/program:release_manager",
    ],
)
