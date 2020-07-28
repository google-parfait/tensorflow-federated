package(default_visibility = ["//visibility:public"])

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
    name = "test",
    testonly = True,
    srcs = ["test.py"],
    srcs_version = "PY3",
)

py_test(
    name = "test_test",
    size = "small",
    srcs = ["test_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":test"],
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
