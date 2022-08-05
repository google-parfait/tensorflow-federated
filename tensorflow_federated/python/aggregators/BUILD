load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":aggregators_packages",
    "//tensorflow_federated/python/analytics:analytics_packages",
    "//tensorflow_federated/python/learning:learning_packages",
    "//tensorflow_federated/python/simulation:simulation_packages",
])

package_group(
    name = "aggregators_packages",
    packages = [
        "//tensorflow_federated/python/aggregators/...",
    ],
)

licenses(["notice"])

py_library(
    name = "aggregators",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":concat",
        ":differential_privacy",
        ":elias_gamma_encode",
        ":encoded",
        ":factory",
        ":factory_utils",
        ":mean",
        ":measurements",
        ":primitives",
        ":quantile_estimation",
        ":robust",
        ":rotation",
        ":sampling",
        ":secure",
        ":sum_factory",
    ],
)

py_library(
    name = "aggregator_test_utils",
    testonly = True,
    srcs = ["aggregator_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "aggregator_test_utils_test",
    size = "small",
    srcs = ["aggregator_test_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregator_test_utils",
        ":factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "concat",
    srcs = ["concat.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "concat_test",
    srcs = ["concat_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":concat",
        ":mean",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "deterministic_discretization",
    srcs = ["deterministic_discretization.py"],
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "deterministic_discretization_test",
    srcs = ["deterministic_discretization_test.py"],
    deps = [
        ":deterministic_discretization",
        ":mean",
        ":measurements",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "differential_privacy",
    srcs = ["differential_privacy.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "differential_privacy_test",
    timeout = "long",
    srcs = ["differential_privacy_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregator_test_utils",
        ":differential_privacy",
        ":factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "discretization",
    srcs = ["discretization.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "discretization_test",
    srcs = ["discretization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":discretization",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "distributed_dp",
    srcs = ["distributed_dp.py"],
    srcs_version = "PY3",
    deps = [
        ":concat",
        ":differential_privacy",
        ":discretization",
        ":factory",
        ":modular_clipping",
        ":quantile_estimation",
        ":robust",
        ":rotation",
        ":secure",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "distributed_dp_test",
    timeout = "eternal",
    srcs = ["distributed_dp_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":concat",
        ":differential_privacy",
        ":discretization",
        ":distributed_dp",
        ":factory",
        ":robust",
        ":rotation",
        ":secure",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/core/test:static_assert",
    ],
)

py_library(
    name = "elias_gamma_encode",
    srcs = ["elias_gamma_encode.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/aggregators:concat",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "elias_gamma_encode_test",
    size = "medium",
    srcs = ["elias_gamma_encode_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":elias_gamma_encode",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "encoded",
    srcs = ["encoded.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "encoded_test",
    size = "medium",
    srcs = ["encoded_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":encoded",
        ":factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "factory",
    srcs = ["factory.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/templates:aggregation_process",
    ],
)

py_library(
    name = "factory_utils",
    srcs = ["factory_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/templates:aggregation_process",
    ],
)

py_test(
    name = "factory_utils_test",
    srcs = ["factory_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":factory_utils",
        "//tensorflow_federated/python/aggregators:factory",
        "//tensorflow_federated/python/aggregators:mean",
        "//tensorflow_federated/python/aggregators:sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "hadamard",
    srcs = ["hadamard.py"],
    srcs_version = "PY3",
)

py_test(
    name = "hadamard_test",
    srcs = ["hadamard_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":hadamard"],
)

py_library(
    name = "mean",
    srcs = ["mean.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "mean_test",
    size = "medium",
    srcs = ["mean_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregator_test_utils",
        ":factory",
        ":mean",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "measurements",
    srcs = ["measurements.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "measurements_test",
    srcs = ["measurements_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":mean",
        ":measurements",
        ":primitives",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "modular_clipping",
    srcs = ["modular_clipping.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        ":sum_factory",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "modular_clipping_test",
    srcs = ["modular_clipping_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":modular_clipping",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "primitives",
    srcs = ["primitives.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/federated_context:value_impl",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_test(
    name = "primitives_test",
    timeout = "long",
    srcs = ["primitives_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":primitives",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/test:static_assert",
    ],
)

py_library(
    name = "quantile_estimation",
    srcs = ["quantile_estimation.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        ":secure",
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:estimation_process",
    ],
)

py_test(
    name = "quantile_estimation_test",
    srcs = ["quantile_estimation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":quantile_estimation",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/test:static_assert",
    ],
)

py_library(
    name = "robust",
    srcs = ["robust.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "robust_test",
    srcs = ["robust_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregator_test_utils",
        ":mean",
        ":robust",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "rotation",
    srcs = ["rotation.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        ":hadamard",
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "rotation_test",
    srcs = ["rotation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":mean",
        ":measurements",
        ":rotation",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_base",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_transformations",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "sampling_test",
    srcs = ["sampling_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sampling",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
    ],
)

py_library(
    name = "secure",
    srcs = ["secure.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        ":primitives",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "secure_test",
    size = "medium",
    srcs = ["secure_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":factory",
        ":secure",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "//tensorflow_federated/python/core/test:static_assert",
    ],
)

py_library(
    name = "stochastic_discretization",
    srcs = ["stochastic_discretization.py"],
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/tensorflow_context:tensorflow_computation",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "stochastic_discretization_test",
    srcs = ["stochastic_discretization_test.py"],
    deps = [
        ":mean",
        ":measurements",
        ":stochastic_discretization",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "sum_factory",
    srcs = ["sum_factory.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/federated_context:federated_computation",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "sum_factory_test",
    size = "small",
    srcs = ["sum_factory_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":factory",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)
