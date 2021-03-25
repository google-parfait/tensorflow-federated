load("@rules_python//python:defs.bzl", "py_library", "py_test")

# Packages that have visibility into the `simulation` package.
package_group(name = "simulation_visibility")

package_group(
    name = "default_visibility",
    includes = [":simulation_visibility"],
    packages = [
        # Subpackages
        "//tensorflow_federated/python/simulation/...",

        # Core Dependencies
        "//tensorflow_federated/python/core/test/...",
    ],
)

package(default_visibility = [":default_visibility"])

licenses(["notice"])

py_library(
    name = "simulation",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":checkpoint_manager",
        ":client_data",
        ":csv_manager",
        ":file_per_user_client_data",
        ":hdf5_client_data",
        ":iterative_process_compositions",
        ":metrics_manager",
        ":sampling_utils",
        ":server_utils",
        ":sql_client_data",
        ":tensorboard_manager",
        ":training_loop",
        ":transforming_client_data",
        "//tensorflow_federated/python/simulation/datasets",
        "//tensorflow_federated/python/simulation/models",
    ],
)

py_library(
    name = "checkpoint_manager",
    srcs = ["checkpoint_manager.py"],
    srcs_version = "PY3",
)

py_test(
    name = "checkpoint_manager_test",
    srcs = ["checkpoint_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":checkpoint_manager"],
)

py_library(
    name = "client_data",
    srcs = ["client_data.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
    ],
)

py_test(
    name = "client_data_test",
    size = "small",
    srcs = ["client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":client_data"],
)

py_library(
    name = "csv_manager",
    srcs = ["csv_manager.py"],
    srcs_version = "PY3",
    deps = [":metrics_manager"],
)

py_test(
    name = "csv_manager_test",
    srcs = ["csv_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":csv_manager"],
)

py_library(
    name = "file_per_user_client_data",
    srcs = ["file_per_user_client_data.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/simulation/datasets:file_per_user_client_data"],
)

py_test(
    name = "file_per_user_client_data_test",
    size = "small",
    srcs = ["file_per_user_client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":file_per_user_client_data"],
)

py_library(
    name = "from_tensor_slices_client_data",
    srcs = ["from_tensor_slices_client_data.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/simulation/datasets:from_tensor_slices_client_data",
    ],
)

py_test(
    name = "from_tensor_slices_client_data_test",
    size = "small",
    srcs = ["from_tensor_slices_client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":from_tensor_slices_client_data",
        "//tensorflow_federated/python/core/api:test_case",
    ],
)

py_library(
    name = "hdf5_client_data",
    srcs = ["hdf5_client_data.py"],
    deprecation = "hdf5_client_data is deprecated and slated for removal April 2021",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_test(
    name = "hdf5_client_data_test",
    size = "small",
    srcs = ["hdf5_client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":hdf5_client_data"],
)

py_library(
    name = "iterative_process_compositions",
    srcs = ["iterative_process_compositions.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
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
        "//tensorflow_federated/python/core/api:computation_types",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:intrinsics",
        "//tensorflow_federated/python/core/impl/types:placement_literals",
        "//tensorflow_federated/python/core/templates:iterative_process",
    ],
)

py_library(
    name = "metrics_manager",
    srcs = ["metrics_manager.py"],
    srcs_version = "PY3",
)

py_library(
    name = "sampling_utils",
    srcs = ["sampling_utils.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated/python/simulation/datasets:client_data"],
)

py_test(
    name = "sampling_utils_test",
    size = "small",
    srcs = ["sampling_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sampling_utils",
        "//tensorflow_federated/python/simulation/datasets:client_data",
    ],
)

py_library(
    name = "server_utils",
    srcs = ["server_utils.py"],
    srcs_version = "PY3",
    deps = [
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
    name = "tensorboard_manager",
    srcs = ["tensorboard_manager.py"],
    srcs_version = "PY3",
    deps = [":metrics_manager"],
)

py_test(
    name = "tensorboard_manager_test",
    srcs = ["tensorboard_manager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tensorboard_manager"],
)

py_library(
    name = "transforming_client_data",
    srcs = ["transforming_client_data.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/simulation/datasets:client_data",
        "//tensorflow_federated/python/simulation/datasets:transforming_client_data",
    ],
)

py_test(
    name = "transforming_client_data_test",
    size = "small",
    srcs = ["transforming_client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":transforming_client_data",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/simulation/datasets:from_tensor_slices_client_data",
    ],
)

py_library(
    name = "sql_client_data",
    srcs = ["sql_client_data.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/simulation/datasets:sql_client_data",
    ],
)

py_test(
    name = "sql_client_data_test",
    size = "small",
    srcs = ["sql_client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":sql_client_data"],
)

py_library(
    name = "training_loop",
    srcs = ["training_loop.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/simulation:checkpoint_manager",
        "//tensorflow_federated/python/simulation:metrics_manager",
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
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/simulation:checkpoint_manager",
        "//tensorflow_federated/python/simulation:metrics_manager",
    ],
)
