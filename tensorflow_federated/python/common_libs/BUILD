load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//tensorflow_federated:internal"])

licenses(["notice"])

# Note: TFF code outside this directory can depend on libraries defined in this
# directory, but not the other way round. In particular, utility classes and
# functions that depend on interfaces defined in the Core API belong in `core`,
# and those that deal with TensorFlow belong in `tensorflow_libs`.

py_library(
    name = "common_libs",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
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
        "$(location golden_test_goldens/test_check_string_fails.expected)",
        "--golden",
        "$(location golden_test_goldens/test_check_string_succeeds.expected)",
        "--golden",
        "$(location golden_test_goldens/test_check_string_updates.expected)",
    ],
    data = [
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
    # TODO(b/163856734): Fix the visibility of the `common_libs`.
    visibility = ["//tensorflow_federated:internal"],
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
    name = "serialization_utils",
    srcs = ["serialization_utils.py"],
    srcs_version = "PY3",
    # TODO(b/163856734): Fix the visibility of the `common_libs`.
    visibility = ["//tensorflow_federated:internal"],
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
    # TODO(b/163856734): Fix the visibility of the `common_libs`.
    visibility = [
        "//tensorflow_federated:internal",
        "//tensorflow_federated/python/tests:__pkg__",
    ],
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
    srcs = ["test_utils.py"],
    srcs_version = "PY3",
    # TODO(b/163856734): Fix the visibility of the `common_libs`.
    visibility = [
        "//tensorflow_federated:internal",
        "//tensorflow_federated/python/examples/simple_fedavg:__pkg__",
        "//tensorflow_federated/python/tests:__pkg__",
    ],
)

py_library(
    name = "tracing",
    srcs = ["tracing.py"],
    srcs_version = "PY3",
    # TODO(b/163856734): Fix the visibility of the `common_libs`.
    visibility = ["//tensorflow_federated:internal"],
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
