load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":aggregators_packages",
        "//tensorflow_federated/python/analytics:analytics_packages",
        "//tensorflow_federated/python/learning:learning_packages",
        "//tensorflow_federated/python/simulation:simulation_packages",
    ],
)

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
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":concat",
        ":differential_privacy",
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
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "aggregator_test_utils_test",
    size = "small",
    srcs = ["aggregator_test_utils_test.py"],
    deps = [
        ":aggregator_test_utils",
        ":factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "concat",
    srcs = ["concat.py"],
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "concat_test",
    srcs = ["concat_test.py"],
    deps = [
        ":concat",
        ":mean",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "deterministic_discretization",
    srcs = ["deterministic_discretization.py"],
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
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
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "differential_privacy",
    srcs = ["differential_privacy.py"],
    deps = [
        ":factory",
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "differential_privacy_test",
    timeout = "long",
    srcs = ["differential_privacy_test.py"],
    deps = [
        ":aggregator_test_utils",
        ":differential_privacy",
        ":factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "discretization",
    srcs = ["discretization.py"],
    deps = [
        ":factory",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "discretization_test",
    srcs = ["discretization_test.py"],
    deps = [
        ":discretization",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "factory",
    srcs = ["factory.py"],
    deps = [
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "factory_utils",
    srcs = ["factory_utils.py"],
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "factory_utils_test",
    srcs = ["factory_utils_test.py"],
    deps = [
        ":factory",
        ":factory_utils",
        ":mean",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "hadamard",
    srcs = ["hadamard.py"],
)

py_test(
    name = "hadamard_test",
    srcs = ["hadamard_test.py"],
    deps = [":hadamard"],
)

py_library(
    name = "mean",
    srcs = ["mean.py"],
    deps = [
        ":factory",
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "mean_test",
    size = "medium",
    srcs = ["mean_test.py"],
    deps = [
        ":aggregator_test_utils",
        ":factory",
        ":mean",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "measurements",
    srcs = ["measurements.py"],
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "measurements_test",
    srcs = ["measurements_test.py"],
    deps = [
        ":mean",
        ":measurements",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "modular_clipping",
    srcs = ["modular_clipping.py"],
    deps = [
        ":factory",
        ":sum_factory",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "modular_clipping_test",
    srcs = ["modular_clipping_test.py"],
    deps = [
        ":modular_clipping",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "primitives",
    srcs = ["primitives.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "primitives_test",
    timeout = "long",
    srcs = ["primitives_test.py"],
    deps = [
        ":primitives",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "quantile_estimation",
    srcs = ["quantile_estimation.py"],
    deps = [
        ":factory",
        ":secure",
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "quantile_estimation_test",
    srcs = ["quantile_estimation_test.py"],
    deps = [
        ":quantile_estimation",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "robust",
    srcs = ["robust.py"],
    deps = [
        ":factory",
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "robust_test",
    srcs = ["robust_test.py"],
    deps = [
        ":aggregator_test_utils",
        ":mean",
        ":robust",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "rotation",
    srcs = ["rotation.py"],
    deps = [
        ":factory",
        ":hadamard",
        ":sum_factory",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "rotation_test",
    srcs = ["rotation_test.py"],
    deps = [
        ":mean",
        ":measurements",
        ":rotation",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "sampling_test",
    srcs = ["sampling_test.py"],
    deps = [
        ":sampling",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "secure",
    srcs = ["secure.py"],
    deps = [
        ":factory",
        ":primitives",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/backends/mapreduce:intrinsics",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "secure_test",
    size = "medium",
    srcs = ["secure_test.py"],
    deps = [
        ":factory",
        ":secure",
        "//tensorflow_federated/python/core/backends/test:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:estimation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "stochastic_discretization",
    srcs = ["stochastic_discretization.py"],
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
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
        "//tensorflow_federated/python/core/environments/tensorflow_backend:type_conversions",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "sum_factory",
    srcs = ["sum_factory.py"],
    deps = [
        ":factory",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "sum_factory_test",
    size = "small",
    srcs = ["sum_factory_test.py"],
    deps = [
        ":factory",
        ":sum_factory",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/templates:aggregation_process",
        "//tensorflow_federated/python/core/templates:measured_process",
        "@federated_language//federated_language",
    ],
)
