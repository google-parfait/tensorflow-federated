load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":aggregators_packages",
    ":aggregators_visibility",
])

package_group(
    name = "aggregators_packages",
    packages = ["//tensorflow_federated/python/aggregators/..."],
)

package_group(
    name = "aggregators_visibility",
    includes = [
        "//tensorflow_federated/python/analytics:analytics_packages",
        "//tensorflow_federated/python/learning:learning_packages",
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
        ":encoded",
        ":factory",
        ":mean",
        ":measurements",
        ":primitives",
        ":quantile_estimation",
        ":robust",
        ":sampling",
        ":secure",
        ":sum_factory",
    ],
)

py_library(
    name = "concat",
    srcs = ["concat.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
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
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
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
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
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
        ":differential_privacy",
        ":factory",
        ":test_utils",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
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
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
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
        "//tensorflow_federated/python/core/api:test_case",
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
    name = "mean",
    srcs = ["mean.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
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
        ":factory",
        ":mean",
        ":test_utils",
        "//tensorflow_federated/python/core/api:test_case",
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
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
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
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
    ],
)

py_library(
    name = "primitives",
    srcs = ["primitives.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/federated_context:value_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
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
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
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
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
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
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/templates:estimation_process",
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
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
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
        ":mean",
        ":robust",
        ":sum_factory",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
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
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
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
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
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
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/api:computation_base",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
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
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
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
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
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
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
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
        "//tensorflow_federated/python/core/api:computations",
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
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/api:computations",
        "//tensorflow_federated/python/core/impl/federated_context:intrinsics",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)

py_test(
    name = "test_utils_test",
    size = "small",
    srcs = ["test_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":factory",
        ":test_utils",
        "//tensorflow_federated/python/core/api:test_case",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
    ],
)
