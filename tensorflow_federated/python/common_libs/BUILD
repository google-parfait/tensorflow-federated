load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    ":common_libs_packages",
    "//tensorflow_federated/python:internal_packages",

    # Namespace visibility
    "//tensorflow_federated:__pkg__",
])

package_group(
    name = "common_libs_packages",
    packages = ["//tensorflow_federated/python/common_libs/..."],
)

licenses(["notice"])

py_library(
    name = "common_libs",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools/python_package:python_package_tool"],
)

py_library(
    name = "golden",
    testonly = True,
    srcs = ["golden.py"],
    srcs_version = "PY3",
)

py_test(
    name = "golden_test",
    size = "small",
    srcs = ["golden_test.py"],
    args = [
        "--golden",
        "$(location golden_test_goldens/test_check_raises_traceback.expected)",
        "--golden",
        "$(location golden_test_goldens/test_check_string_fails.expected)",
        "--golden",
        "$(location golden_test_goldens/test_check_string_succeeds.expected)",
        "--golden",
        "$(location golden_test_goldens/test_check_string_updates.expected)",
    ],
    data = [
        "golden_test_goldens/test_check_raises_traceback.expected",
        "golden_test_goldens/test_check_string_fails.expected",
        "golden_test_goldens/test_check_string_succeeds.expected",
        "golden_test_goldens/test_check_string_updates.expected",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":golden"],
)

py_library(
    name = "py_typecheck",
    srcs = ["py_typecheck.py"],
    srcs_version = "PY3",
)

py_test(
    name = "py_typecheck_test",
    size = "small",
    srcs = ["py_typecheck_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":py_typecheck",
        ":structure",
    ],
)

py_library(
    name = "retrying",
    srcs = ["retrying.py"],
    srcs_version = "PY3",
    deps = [
        ":py_typecheck",
    ],
)

py_test(
    name = "retrying_test",
    size = "small",
    srcs = ["retrying_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":retrying"],
)

py_library(
    name = "serialization_utils",
    srcs = ["serialization_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":py_typecheck",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "serialization_utils_test",
    size = "small",
    srcs = ["serialization_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":serialization_utils",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "structure",
    srcs = ["structure.py"],
    srcs_version = "PY3",
    deps = [":py_typecheck"],
)

py_test(
    name = "structure_test",
    size = "small",
    srcs = ["structure_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":structure"],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "tracing",
    srcs = ["tracing.py"],
    srcs_version = "PY3",
    deps = [":py_typecheck"],
)

py_test(
    name = "tracing_test",
    size = "small",
    srcs = ["tracing_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tracing"],
)
