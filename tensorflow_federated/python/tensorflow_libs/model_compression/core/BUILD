package(default_visibility = ["//tensorflow_federated/python/tensorflow_libs/model_compression:__subpackages__"])

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
    size = "small",
    srcs = ["encoding_stage_test.py"],
    deps = [
        ":encoding_stage",
        ":test_utils",
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
        "@six_archive//:six",
    ],
)

py_test(
    name = "test_utils_test",
    size = "small",
    srcs = ["test_utils_test.py"],
    deps = [
        ":encoding_stage",
        ":test_utils",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":encoding_stage",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
