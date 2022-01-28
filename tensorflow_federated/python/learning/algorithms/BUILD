load("//tensorflow_federated/tools:build_defs.bzl", "py_cpu_gpu_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":algorithms_packages",
    "//tensorflow_federated/python/learning:learning_visibility",
])

package_group(
    name = "algorithms_packages",
    packages = ["//tensorflow_federated/python/learning/algorithms/..."],
)

licenses(["notice"])

py_library(
    name = "algorithms",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":fed_avg",
        ":fed_avg_with_optimizer_schedule",
        ":fed_prox",
        ":fed_sgd",
        ":kmeans_clustering",
    ],
)

py_library(
    name = "aggregation",
    srcs = ["aggregation.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/templates:aggregation_process",
    ],
)

py_test(
    name = "aggregation_test",
    srcs = ["aggregation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregation",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "fed_avg",
    srcs = ["fed_avg.py"],
    srcs_version = "PY3",
    deps = [
        ":aggregation",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:distributors",
        "//tensorflow_federated/python/learning/templates:finalizers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/learning/templates:model_delta_client_work",
    ],
)

py_cpu_gpu_test(
    name = "fed_avg_test",
    timeout = "long",
    srcs = ["fed_avg_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":aggregation",
        ":fed_avg",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/learning:model_examples",
        "//tensorflow_federated/python/learning:model_update_aggregator",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "//tensorflow_federated/python/learning/templates:distributors",
    ],
)

py_library(
    name = "fed_avg_with_optimizer_schedule",
    srcs = ["fed_avg_with_optimizer_schedule.py"],
    srcs_version = "PY3",
    deps = [
        ":fed_avg",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/learning/templates:client_works",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:distributors",
        "//tensorflow_federated/python/learning/templates:finalizers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/learning/templates:model_delta_client_work",
    ],
)

py_cpu_gpu_test(
    name = "fed_avg_with_optimizer_schedule_test",
    timeout = "long",
    srcs = ["fed_avg_with_optimizer_schedule_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":fed_avg_with_optimizer_schedule",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/learning:model_examples",
        "//tensorflow_federated/python/learning:model_update_aggregator",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
    ],
)

py_library(
    name = "fed_prox",
    srcs = ["fed_prox.py"],
    srcs_version = "PY3",
    deps = [
        ":aggregation",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/learning:client_weight_lib",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:distributors",
        "//tensorflow_federated/python/learning/templates:finalizers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/learning/templates:model_delta_client_work",
    ],
)

py_cpu_gpu_test(
    name = "fed_prox_test",
    timeout = "long",
    srcs = ["fed_prox_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":aggregation",
        ":fed_prox",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:iterative_process",
        "//tensorflow_federated/python/learning:model_examples",
        "//tensorflow_federated/python/learning:model_update_aggregator",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/optimizers:sgdm",
        "//tensorflow_federated/python/learning/templates:distributors",
    ],
)

py_library(
    name = "fed_sgd",
    srcs = ["fed_sgd.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning:model",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
        "//tensorflow_federated/python/learning/metrics:aggregator",
        "//tensorflow_federated/python/learning/optimizers:optimizer",
        "//tensorflow_federated/python/learning/templates:client_works",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:distributors",
        "//tensorflow_federated/python/learning/templates:finalizers",
        "//tensorflow_federated/python/learning/templates:learning_process",
        "//tensorflow_federated/python/tensorflow_libs:tensor_utils",
    ],
)

py_cpu_gpu_test(
    name = "fed_sgd_test",
    timeout = "long",
    srcs = ["fed_sgd_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":fed_sgd",
        "//tensorflow_federated/python/common_libs:test_utils",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/learning:model_examples",
        "//tensorflow_federated/python/learning:model_update_aggregator",
        "//tensorflow_federated/python/learning:model_utils",
        "//tensorflow_federated/python/learning/framework:dataset_reduce",
    ],
)

py_library(
    name = "kmeans_clustering",
    srcs = ["kmeans_clustering.py"],
    srcs_version = "PY3",
    deps = [
        ":aggregation",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/learning/templates:client_works",
        "//tensorflow_federated/python/learning/templates:composers",
        "//tensorflow_federated/python/learning/templates:distributors",
        "//tensorflow_federated/python/learning/templates:finalizers",
        "//tensorflow_federated/python/learning/templates:learning_process",
    ],
)

py_test(
    name = "kmeans_clustering_test",
    srcs = ["kmeans_clustering_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":kmeans_clustering",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)
