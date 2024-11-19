load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":templates_packages",
        "//tensorflow_federated/python/core:core_users",
        "//tensorflow_federated/python/core/backends:backends_packages",
    ],
)

package_group(
    name = "templates_packages",
    packages = ["//tensorflow_federated/python/core/templates/..."],
)

licenses(["notice"])

py_library(
    name = "templates",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated:__pkg__"],
    deps = [
        ":aggregation_process",
        ":estimation_process",
        ":iterative_process",
        ":measured_process",
    ],
)

py_library(
    name = "aggregation_process",
    srcs = ["aggregation_process.py"],
    deps = [
        ":errors",
        ":measured_process",
        "//tensorflow_federated/python/common_libs:structure",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "aggregation_process_test",
    size = "small",
    srcs = ["aggregation_process_test.py"],
    deps = [
        ":aggregation_process",
        ":errors",
        ":measured_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
)

py_library(
    name = "estimation_process",
    srcs = ["estimation_process.py"],
    deps = [
        ":errors",
        ":iterative_process",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "estimation_process_test",
    size = "small",
    srcs = ["estimation_process_test.py"],
    deps = [
        ":errors",
        ":estimation_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "iterative_process",
    srcs = ["iterative_process.py"],
    deps = [
        ":errors",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "iterative_process_test",
    size = "small",
    srcs = ["iterative_process_test.py"],
    deps = [
        ":errors",
        ":iterative_process",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "measured_process",
    srcs = ["measured_process.py"],
    deps = [
        ":errors",
        ":iterative_process",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "measured_process_test",
    size = "small",
    srcs = ["measured_process_test.py"],
    args = [
        "--golden",
        "$(location measured_process_test_goldens/composition_with_aggregation_processes.expected)",
        "--golden",
        "$(location measured_process_test_goldens/composition_with_measured_processes.expected)",
        "--golden",
        "$(location measured_process_test_goldens/concatenation_with_measured_processes.expected)",
        "--golden",
        "$(location measured_process_test_goldens/concatenation_with_aggregation_processes.expected)",
        "--verbose",
    ],
    data = [
        "measured_process_test_goldens/composition_with_aggregation_processes.expected",
        "measured_process_test_goldens/composition_with_measured_processes.expected",
        "measured_process_test_goldens/concatenation_with_aggregation_processes.expected",
        "measured_process_test_goldens/concatenation_with_measured_processes.expected",
    ],
    deps = [
        ":aggregation_process",
        ":errors",
        ":iterative_process",
        ":measured_process",
        "//tensorflow_federated/python/core/impl/compiler:compiler_test_utils",
        "@federated_language//federated_language",
    ],
)
