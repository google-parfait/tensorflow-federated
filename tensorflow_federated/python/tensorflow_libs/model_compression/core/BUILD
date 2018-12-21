# Placeholder for internal Python version compatibility macro.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "encoding_stage",
    srcs = ["encoding_stage.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "encoding_stage_test",
    srcs = ["encoding_stage_test.py"],
    default_python_version = "PY3",
    deps = [
        ":encoding_stage",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    deps = [
        ":encoding_stage",
        ":utils",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    default_python_version = "PY3",
    deps = [
        ":test_utils",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    default_python_version = "PY3",
    deps = [
        ":utils",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
